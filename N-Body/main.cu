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

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
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

    glm::vec3 at(0,0,0);
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

__global__ void nBodySimulation(Particle* currentParticles, Particle* lastParticles, float dt, int numParticles) {
    // Get the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numParticles) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        float cx = currentParticles[tid].x;
        float cy = currentParticles[tid].y;
        float cz = currentParticles[tid].z;

        // Compute acceleration
        for (int j = 0; j < numParticles; j++) {
            float dx = lastParticles[j].x - cx;
            float dy = lastParticles[j].y - cy;
            float dz = lastParticles[j].z - cz;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            //float invDist = rsqrtf(distSqr);
            float invDist = sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            float force = lastParticles[j].mass * invDist3;
            //float force = 1 * invDist3;

            ax += dx * force;
            ay += dy * force;
            az += dz * force;
        }

        // Update velocity
        currentParticles[tid].vx += ax * dt;
        currentParticles[tid].vy += ay * dt;
        currentParticles[tid].vz += az * dt;

        // Update position
        currentParticles[tid].x += currentParticles[tid].vx * dt;
        currentParticles[tid].y += currentParticles[tid].vy * dt;
        currentParticles[tid].z += currentParticles[tid].vz * dt;
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
void InitVaoVbo()
{
    // 1 db VAO foglalása
    glGenVertexArrays(1, &vaoID);
    // a frissen generált VAO beállítasa aktívnak
    glBindVertexArray(vaoID);

    // hozzunk létre egy új VBO erõforrás nevet
    glGenBuffers(1, &vboID);
    glBindBuffer(GL_ARRAY_BUFFER, vboID);

    // VAO-ban jegyezzük fel, hogy a VBO-ban az elsõ 3 float sizeof(Particle)-enként lesz az elsõ attribútum (pozíció)
    glEnableVertexAttribArray(0); // ez lesz majd a pozíció
    glVertexAttribPointer(
        (GLuint)0,				// a VB-ben található adatok közül a 0. "indexû" attribútumait állítjuk be
        3,				// komponens szám
        GL_FLOAT,		// adatok típusa
        GL_FALSE,		// normalizált legyen-e
        sizeof(Particle),	// stride (0=egymás után)
        0				// a 0. indexû attribútum hol kezdõdik a sizeof(Particle)-nyi területen belül
    );

    // a második attribútumhoz pedig a VBO-ban sizeof(Particle) ugrás után sizeof(glm::vec3)-nyit menve újabb 3 float adatot találunk (gyorsulás)
    glEnableVertexAttribArray(1); // ez lesz majd a gyorsulás
    glVertexAttribPointer(
        (GLuint)1,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Particle),
        (void*)(sizeof(glm::vec3)));
    // a harmadik attribútumhoz pedig a VBO-ban sizeof(Particle) ugrás után 2 * sizeof(glm::vec3)-nyit menve újabb 1 float adatot találunk (tömeg)
    glEnableVertexAttribArray(2); // ez lesz majd a tömeg
    glVertexAttribPointer(
        (GLuint)2,
        1,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Particle),
        (void*)(sizeof(glm::vec3) * 2));

    glBindVertexArray(0); // feltöltüttük a VAO-t, kapcsoljuk le
    glBindBuffer(GL_ARRAY_BUFFER, 0); // feltöltöttük a VBO-t is, ezt is vegyük le
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

int numParticles = BLOCK_SIZE * 40;
int numIterations = 20000;
float dt = 0.0003f;
Particle* particlesHost = (Particle*)malloc(numParticles * sizeof(Particle));
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

void Render(glm::mat4 mvp)
{
    glBindBuffer(GL_ARRAY_BUFFER, vboID); // tegyük "aktívvá" a létrehozott VBO-t
    // töltsük fel adatokkal az aktív VBO-t
    glBufferData(GL_ARRAY_BUFFER,					// az aktív VBO-ba töltsünk adatokat
        sizeof(Particle) * numParticles,		// ennyi bájt nagyságban
        particlesHost,								// errõl a rendszermemóriabeli címrõl olvasva
        //GL_STATIC_DRAW);					// úgy, hogy a VBO-nkba nem tervezünk ezután írni és minden kirajzoláskor felhasnzáljuk a benne lévõ adatokat	
        GL_STREAM_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // shader bekapcsolása, ebben a projektben a teljes programot jelöli, hisz nem váltunk a shaderek között
    glUseProgram(shaderProgramID);

    // kapcsoljuk be a VAO-t (a VBO jön vele együtt)
    glBindVertexArray(vaoID);

    // uniformok
    glm::mat4 locMVP = mvp;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgramID, "MVP"), 1, GL_FALSE, &locMVP[0][0]);

    glDrawArrays(GL_POINTS, 0, numParticles);

    // VAO kikapcsolása
    glBindVertexArray(0);

    // shader kikapcsolása
    glUseProgram(0);
}

void display(GLFWwindow* window, Particle* particles, int numParticles, glm::mat4 projection, glm::mat4 view) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Render(projection * view);

    SDL_GL_SwapWindow(win);
}

int main(int argc, char* argv[]) {
    bool quit = false;

    InitParticles();
    Particle* particlesDev[2];
    size_t particlesSize = numParticles * sizeof(Particle);
    cudaMalloc((void**)&particlesDev[0], particlesSize);
    cudaMalloc((void**)&particlesDev[1], particlesSize);

    // Copy initial particle positions to device
    cudaMemcpy(particlesDev[0], particlesHost, particlesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(particlesDev[1], particlesHost, particlesSize, cudaMemcpyHostToDevice);

    InitSDLWindow();

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    RotateCamera(0, 0);

    InitRendering();

    // Start the simulation loop
    for (int i = 0; i < numIterations; i++) {
        if (quit)
        {
            break;
        }

        // Swap particle buffers
        int currentBuffer = i % 2;
        int nextBuffer = (i + 1) % 2;

        // Launch the kernel
        nBodySimulation << <(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (particlesDev[currentBuffer], particlesDev[nextBuffer], dt, numParticles);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // Display particles
        cudaMemcpy(particlesHost, particlesDev[nextBuffer], particlesSize, cudaMemcpyDeviceToHost);
        //display(window, particlesHost, numParticles, projection, view);
        display(0, particlesHost, numParticles, projection, view);

        while (SDL_PollEvent(&event))
        {
            HandleEvents(event, quit);
        }
    }

    // Cleanup
    glfwTerminate();
    cudaFree(particlesDev[0]);
    cudaFree(particlesDev[1]);
    free(particlesHost);

    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(win);

    return 0;
}