#define GLEW_STATIC

#include <Windows.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/GLUtils.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <SDL.h>
#include <SDL_opengl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <iostream>
#include <corecrt_math_defines.h>

#define SOFTENING 1e-9f
#define BLOCK_SIZE 128
#define THETA 1.0f

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};
struct Body {
    float x, y, z;
    float mass;
};
struct Node {
    float x, y, z;
    float mass;
    float cx, cy, cz;
    Node* children[8];
};


// Declare global variables for mouse movement
int lastMouseX = 0;
int lastMouseY = 0;
bool mousePressed = false;
glm::mat4 view; // View matrix
float mouseAngleX = 0.0f;
float mouseAngleY = 0.8f;
float rotationSpeed = 0.01f;
float cameraDistance = 2;

glm::vec3 camUp;
glm::vec3 camForward;
glm::vec3 camRight;

int numParticles = BLOCK_SIZE * 80;
int numIterations = 20000;
float dt = 0.0003f;
Particle* particlesHost = (Particle*)malloc(numParticles * sizeof(Particle));

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator*(float a, const float3& v) {
    return make_float3(a * v.x, a * v.y, a * v.z);
}
__device__ float3 operator*(const float3& v, float a) {
    return make_float3(a * v.x, a * v.y, a * v.z);
}

__device__ float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
void RotateCamera(int dx, int dy)
{
    mouseAngleX += dx * rotationSpeed;
    mouseAngleY += dy * rotationSpeed;
    if (mouseAngleY > 2.8)
    {
        mouseAngleY = 2.8;
    }
    else if (mouseAngleY < 0.2)
    {
        mouseAngleY = 0.2;
    }

    glm::vec3 at(0, 0, 0);
    glm::vec3 eye = at;
    eye.x += cameraDistance;
    glm::vec3 up(0, 0, 1);

    camForward = glm::normalize(at - eye);
    camRight = glm::normalize(glm::cross(camForward, up));
    camUp = glm::normalize(glm::cross(camRight, camForward));

    eye = at + cameraDistance * glm::vec3(cosf(mouseAngleX) * sinf(mouseAngleY),
        sinf(mouseAngleX) * sinf(mouseAngleY),
        cosf(mouseAngleY));

    view = glm::lookAt(eye, at, up);
}

__device__ float3 computeGravity(float3 posA, float massA, float3 posB) {
    float3 dir = posB - posA;
    float distSqr = dot(dir, dir) + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;
    return massA * invDist3 * dir;
}

__device__ void computeCenterOfMass(Particle* particles, int start, int end, float3& center, float& mass) {
    center = make_float3(0.0f, 0.0f, 0.0f);
    mass = 0.0f;

    for (int i = start; i < end; i++) {
        center = center + make_float3(particles[i].x, particles[i].y, particles[i].z) * particles[i].mass;
        mass += particles[i].mass;
    }

    center = (mass > 0.0f) ? center * (1.0f / mass) : make_float3(0.0f, 0.0f, 0.0f);
}
__device__ void computeCenterOfMass(Node* particles, int start, int end, float3& center, float& mass) {
    center = make_float3(0.0f, 0.0f, 0.0f);
    mass = 0.0f;

    for (int i = start; i < end; i++) {
        center = center + make_float3(particles[i].x, particles[i].y, particles[i].z) * particles[i].mass;
        mass += particles[i].mass;
    }

    center = (mass > 0.0f) ? center * (1.0f / mass) : make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void buildQuadtree(Particle* particles, Node* nodes, int numParticles, int maxDepth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numParticles) {
        Node* root = nodes + maxDepth * blockDim.x * gridDim.x;

        float3 center;
        float mass;
        computeCenterOfMass(particles, tid, tid + 1, center, mass);

        int nodeIndex = tid + maxDepth * blockDim.x * blockIdx.x;
        Node* node = nodes + nodeIndex;
        node->x = center.x;
        node->y = center.y;
        node->z = center.z;
        node->mass = mass;

        for (int level = maxDepth - 1; level >= 0; level--) {
            int parentIndex = nodeIndex - blockDim.x;
            Node* parent = nodes + parentIndex;

            if (tid % blockDim.x == 0) {
                parent->children[0] = node;
            }
            else if (tid % blockDim.x == 1) {
                parent->children[1] = node;
            }
            else if (tid % blockDim.x == 2) {
                parent->children[2] = node;
            }
            else if (tid % blockDim.x == 3) {
                parent->children[3] = node;
            }
            else if (tid % blockDim.x == 4) {
                parent->children[4] = node;
            }
            else if (tid % blockDim.x == 5) {
                parent->children[5] = node;
            }
            else if (tid % blockDim.x == 6) {
                parent->children[6] = node;
            }
            else if (tid % blockDim.x == 7) {
                parent->children[7] = node;
            }

            __syncthreads();

            if (tid < blockDim.x) {
                float3 center;
                float mass;
                computeCenterOfMass(nodes + parentIndex, 0, 8, center, mass);

                node->x = center.x;
                node->y = center.y;
                node->z = center.z;
                node->mass = mass;
            }

            __syncthreads();

            nodeIndex = parentIndex;
            node = parent;
        }

        if (tid == 0) {
            Node* root = nodes + maxDepth * blockDim.x * gridDim.x;
            int block = blockDim.x;
            float3 center = make_float3(root->cx, root->cy, root->cz);
            computeCenterOfMass(nodes, 0, block, center, root->mass);
        }
    }
}

__device__ void computeForces(Node* nodes, Particle* particles, int particleIndex, float3& force) {
    force = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < blockDim.x; i++) {
        Node* node = nodes + i + blockDim.x * blockIdx.x;

        float3 posA = make_float3(particles[particleIndex].x, particles[particleIndex].y, particles[particleIndex].z);
        float massA = particles[particleIndex].mass;
        float3 posB = make_float3(node->x, node->y, node->z);
        float massB = node->mass;

        float3 dir = posB - posA;
        float distSqr = dot(dir, dir) + SOFTENING;
        float invDist = rsqrtf(distSqr);

        if (distSqr == 0.0f) {
            continue;
        }

        if (node->mass == 0.0f || (blockDim.x * blockIdx.x + i) == particleIndex) {
            continue;
        }

        if (distSqr < THETA * THETA) {
            force = force + computeGravity(posA, massA, posB);
        }
        else {
            force = force + massB * invDist * invDist * invDist * dir;
        }
    }
}

__global__ void nBodySimulation(Particle* particles, Node* nodes, float dt, int numParticles, int maxDepth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numParticles) {
        float3 force;
        computeForces(nodes, particles, tid, force);

        particles[tid].vx += force.x * dt;
        particles[tid].vy += force.y * dt;
        particles[tid].vz += force.z * dt;
        particles[tid].x += particles[tid].vx * dt;
        particles[tid].y += particles[tid].vy * dt;
        particles[tid].z += particles[tid].vz * dt;
    }
}

SDL_Window* win;
SDL_GLContext context;
SDL_Event event;
void InitSDLWindow()
{
    if (SDL_Init(SDL_INIT_VIDEO) == -1)
    {
        // irjuk ki a hibat es termináljon a program
        std::cout << "[SDL initialization] Error during the SDL initialization: " << SDL_GetError() << std::endl;
        return;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
    // duplapufferelés
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    // mélységi puffer hány bites legyen
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    win = 0;
    win = SDL_CreateWindow("Our City",		// az ablak fejléce
        50,						// az ablak bal-felsõ sarkának kezdeti X koordinátája
        50,						// az ablak bal-felsõ sarkának kezdeti Y koordinátája
        600,						// ablak szélessége
        600,						// és magassága
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

    if (win == 0)
    {
        std::cout << "[Ablak létrehozása]Hiba az SDL inicializálása közben: " << SDL_GetError() << std::endl;
        return;
    }

    context = SDL_GL_CreateContext(win);
    if (context == 0)
    {
        std::cout << "[OGL context creation] Error during the creation of the OGL context: " << SDL_GetError() << std::endl;
        return;
    }

    // 0 - nincs vsync
    // 1 - van vsync
    SDL_GL_SetSwapInterval(1);

    GLenum error = glewInit();
    if (error != GLEW_OK)
    {
        std::cout << "[GLEW] Error during the initialization of glew." << std::endl;
        return;
    }

    int glVersion[2] = { -1, -1 };
    glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]);
    glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]);
    std::cout << "Running OpenGL " << glVersion[0] << "." << glVersion[1] << std::endl;

    if (glVersion[0] == -1 && glVersion[1] == -1)
    {
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(win);

        std::cout << "[OGL context creation] Error during the inialization of the OGL context! Maybe one of the SDL_GL_SetAttribute(...) calls is erroneous." << std::endl;
        return;
    }

    // engedélyezzük és állítsuk be a debug callback függvényt ha debug context-ben vagyunk 
    GLint context_flags;
    glGetIntegerv(GL_CONTEXT_FLAGS, &context_flags);
}

void HandleEvents(SDL_Event ev, bool& quit)
{
    switch (ev.type)
    {
    case SDL_QUIT:
        std::cout << "quit" << std::endl;
        quit = true;
        break;
    case SDL_KEYDOWN:
        //if (ev.key.keysym.sym == SDLK_ESCAPE)
            //quit = true;
        //app->KeyboardDown(ev.key);
        break;
    case SDL_KEYUP:
        //app->KeyboardUp(ev.key);
        break;
    case SDL_MOUSEBUTTONDOWN:
        if (ev.button.button == SDL_BUTTON_LEFT)
        {
            mousePressed = true;
        }
        break;
    case SDL_MOUSEBUTTONUP:
        if (ev.button.button == SDL_BUTTON_LEFT)
        {
            mousePressed = false;
        }
        break;
    case SDL_MOUSEWHEEL:
        if (ev.wheel.y > 0) // scroll up
        {
            cameraDistance -= 0.2f * cameraDistance;
        }
        else if (ev.wheel.y < 0) // scroll down
        {
            cameraDistance += 0.2f * cameraDistance;
        }
        RotateCamera(0, 0);
        break;
    case SDL_MOUSEMOTION:
    {

        int x;
        int y;
        SDL_GetMouseState(&x, &y);

        if (mousePressed)
        {
            int dx = lastMouseX - x;
            int dy = lastMouseY - y;

            RotateCamera(dx, dy);
        }
        lastMouseX = x;
        lastMouseY = y;

        break;
    }

    case SDL_WINDOWEVENT:
        // Néhány platformon (pl. Windows) a SIZE_CHANGED nem hívódik meg az elsõ megjelenéskor.
        // Szerintünk ez bug az SDL könytárban.
        // Ezért ezt az esetet külön lekezeljük, 
        // mivel a MyApp esetlegesen tartalmazhat ablak méret függõ beállításokat, pl. a kamera aspect ratioját a perspective() hívásnál.
        if (ev.window.event == SDL_WINDOWEVENT_SHOWN)
        {
            int w, h;
            SDL_GetWindowSize(win, &w, &h);

            //width = w;
            //height = h;

            //app->Resize(w, h);
        }
        if (ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
        {
            /*
            width = ev.window.data1;
            height = ev.window.data2;
            std::cout << width << " " << height << std::endl;
            app->Resize(ev.window.data1, ev.window.data2);
            */
        }
        break;
    }
}

GLuint vboID;
GLuint vaoID;

GLuint vboPingPong[2];
GLuint vaoPingPong[2];

cudaGraphicsResource* cudaVboResource[2];
void InitVaoVbo()
{
    cudaGLSetGLDevice(0);

    //vao 1
    glGenVertexArrays(1, &vaoPingPong[0]);
    glBindVertexArray(vaoPingPong[0]);

    //vbo 1
    glGenBuffers(1, &vboPingPong[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vboPingPong[0]);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(Particle), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaVboResource[0], vboPingPong[0], cudaGraphicsMapFlagsNone);

    glEnableVertexAttribArray(0);
    //pos
    glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), 0);
    //acc
    glEnableVertexAttribArray(1);
    glVertexAttribPointer((GLuint)1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(sizeof(glm::vec3)));
    //mass
    glEnableVertexAttribArray(2);
    glVertexAttribPointer((GLuint)2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(sizeof(glm::vec3) * 2));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //vao 2
    glGenVertexArrays(1, &vaoPingPong[1]);
    glBindVertexArray(vaoPingPong[1]);

    //vbo 2
    glGenBuffers(1, &vboPingPong[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vboPingPong[1]);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(Particle), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaVboResource[1], vboPingPong[1], cudaGraphicsMapFlagsNone);

    glEnableVertexAttribArray(0);
    //pos
    glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), 0);
    //acc
    glEnableVertexAttribArray(1);
    glVertexAttribPointer((GLuint)1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(sizeof(glm::vec3)));
    //mass
    glEnableVertexAttribArray(2);
    glVertexAttribPointer((GLuint)2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(sizeof(glm::vec3) * 2));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GLuint shaderProgramID;
void InitShaders()
{
    GLuint vs_ID = loadShader(GL_VERTEX_SHADER, "vert.vert");
    GLuint fs_ID = loadShader(GL_FRAGMENT_SHADER, "frag.frag");

    // a shadereket tároló program létrehozása
    shaderProgramID = glCreateProgram();

    // adjuk hozzá a programhoz a shadereket
    glAttachShader(shaderProgramID, vs_ID);
    glAttachShader(shaderProgramID, fs_ID);

    // attributomok osszerendelese a VAO es shader kozt
    glBindAttribLocation(shaderProgramID, 0, "vs_in_pos");
    glBindAttribLocation(shaderProgramID, 1, "vs_in_acc");
    glBindAttribLocation(shaderProgramID, 2, "vs_in_mass");

    // illesszük össze a shadereket (kimenõ-bemenõ változók összerendelése stb.)
    glLinkProgram(shaderProgramID);

    // linkelés ellenõrzese
    GLint infoLogLength = 0, result = 0;

    glGetProgramiv(shaderProgramID, GL_LINK_STATUS, &result);
    glGetProgramiv(shaderProgramID, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (GL_FALSE == result || infoLogLength != 0)
    {
        std::vector<char> VertexShaderErrorMessage(infoLogLength);
        glGetProgramInfoLog(shaderProgramID, infoLogLength, nullptr, VertexShaderErrorMessage.data());

        std::cerr << "[glLinkProgram] Shader linking error:\n" << &VertexShaderErrorMessage[0] << std::endl;
    }

    // már nincs ezekre szükség
    glDeleteShader(vs_ID);
    glDeleteShader(fs_ID);

    //uniformok
    //glUseProgram(shaderProgramID);
}

void InitParticles()
{
    for (int i = 0; i < numParticles; i++) {
        // Generate random angle and distance from the center
        float angle = ((float)rand() / RAND_MAX) * 2 * M_PI;
        float distance = ((float)rand() / RAND_MAX) * 0.5f;

        // Calculate initial positions around the center
        particlesHost[i].x = 0.0f + cos(angle) * distance;
        particlesHost[i].y = 0.0f + sin(angle) * distance;
        particlesHost[i].z = (-0.5f + ((float)rand() / RAND_MAX)) * 0.1f;

        // Calculate initial velocities for rotation
        float speed = distance * distance * 100.0f;
        particlesHost[i].vx = -sin(angle) * speed;
        particlesHost[i].vy = cos(angle) * speed;
        particlesHost[i].vz = 0.0f;

        particlesHost[i].mass = ((float)rand() / RAND_MAX);
    }
}

void InitRendering()
{
    glClearColor(0.125f, 0.25f, 0.5f, 1.0f);

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(2.0);

    InitVaoVbo();
    InitShaders();
}

void RenderParticles(SDL_Window* window, glm::mat4 mvp, int bufferID)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // shader bekapcsolása, ebben a projektben a teljes programot jelöli, hisz nem váltunk a shaderek között
    glUseProgram(shaderProgramID);

    // kapcsoljuk be a VAO-t (a VBO jön vele együtt)
    glBindVertexArray(vaoPingPong[bufferID]);

    // uniformok
    glm::mat4 locMVP = mvp;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgramID, "MVP"), 1, GL_FALSE, &locMVP[0][0]);

    glDrawArrays(GL_POINTS, 0, numParticles);

    // VAO kikapcsolása
    glBindVertexArray(0);

    // shader kikapcsolása
    glUseProgram(0);

    SDL_GL_SwapWindow(win);
}

int main(int argc, char* argv[]) {
    bool quit = false;

    InitParticles();
    Particle* particlesDev[2];
    size_t particlesSize = numParticles * sizeof(Particle);

    InitSDLWindow();

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    RotateCamera(0, 0);

    InitRendering();

    // map cuda
    // get pointer
    // copy data
    // unmap cuda
    size_t size;
    cudaGraphicsMapResources(1, &cudaVboResource[0]);
    cudaGraphicsResourceGetMappedPointer((void**)&particlesDev[0], &size, cudaVboResource[0]);
    cudaMemcpy(particlesDev[0], particlesHost, particlesSize, cudaMemcpyHostToDevice);
    cudaGraphicsUnmapResources(1, &cudaVboResource[0]);

    cudaGraphicsMapResources(1, &cudaVboResource[1]);
    cudaGraphicsResourceGetMappedPointer((void**)&particlesDev[1], &size, cudaVboResource[1]);
    cudaMemcpy(particlesDev[1], particlesHost, particlesSize, cudaMemcpyHostToDevice);
    cudaGraphicsUnmapResources(1, &cudaVboResource[1]);

    // Allocate memory for the quadtree nodes
    int maxDepth = ceil(log2(numParticles));
    size_t nodesSize = (numParticles / BLOCK_SIZE) * maxDepth * sizeof(Node);
    Node* nodesDev;
    cudaMalloc((void**)&nodesDev, nodesSize);

    buildQuadtree << <(numParticles / BLOCK_SIZE), BLOCK_SIZE >> > (particlesDev[0], nodesDev, numParticles, maxDepth);
    // Start the simulation loop
    for (int i = 0; i < numIterations; i++) {
        if (quit)
        {
            break;
        }

        // Swap particle buffers
        int currentBuffer = i % 2;
        int nextBuffer = (i + 1) % 2;

        cudaGraphicsMapResources(1, &cudaVboResource[0]);
        cudaGraphicsResourceGetMappedPointer((void**)&particlesDev[0], &size, cudaVboResource[0]);

        cudaGraphicsMapResources(1, &cudaVboResource[1]);
        cudaGraphicsResourceGetMappedPointer((void**)&particlesDev[1], &size, cudaVboResource[1]);

        // Launch the kernel
        //nBodySimulation << < (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (particlesDev[currentBuffer], particlesDev[nextBuffer], dt, numParticles);
        nBodySimulation << <(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (particlesDev[currentBuffer], nodesDev, dt, numParticles, maxDepth);
        // Wait for kernel to finish
        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &cudaVboResource[0]);
        cudaGraphicsUnmapResources(1, &cudaVboResource[1]);
        // Rebuild the quadtree
        buildQuadtree << <(numParticles / BLOCK_SIZE), BLOCK_SIZE >> > (particlesDev[currentBuffer], nodesDev, numParticles, maxDepth);

        RenderParticles(win, projection * view, currentBuffer);

        while (SDL_PollEvent(&event))
        {
            HandleEvents(event, quit);
        }
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cudaVboResource[0]);
    cudaGraphicsUnregisterResource(cudaVboResource[1]);
    glDeleteBuffers(1, &vboPingPong[0]);
    glDeleteBuffers(1, &vboPingPong[1]);

    free(particlesHost);

    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(win);

    return 0;
}