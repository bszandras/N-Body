#version 130

// VBO-ból érkezõ változók
in vec3 vs_in_pos;
in vec3 vs_in_acc;
in float vs_in_mass;

// a pipeline-ban tovább adandó értékek
out vec3 vs_out_pos;
out vec3 vs_out_acc;
out float vs_out_mass;

uniform mat4 MVP;
uniform vec3 points[256];

void main()
{
	vs_out_pos = vs_in_pos;
	vs_out_acc = vs_in_acc;
	vs_out_mass = vs_in_mass;

	gl_Position = MVP * vec4(vs_in_pos, 1);
}