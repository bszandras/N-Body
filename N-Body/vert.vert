#version 130

// VBO-b�l �rkez� v�ltoz�k
in vec3 vs_in_pos;
in vec3 vs_in_col;

// a pipeline-ban tov�bb adand� �rt�kek
out vec3 vs_out_pos;
out vec3 vs_out_col;

uniform mat4 MVP;
uniform vec3 points[256];

void main()
{
	vs_out_pos = vs_in_pos;
	vs_out_col = vs_in_col;

	gl_Position = MVP * vec4(vs_in_pos, 1);
}