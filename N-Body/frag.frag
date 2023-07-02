#version 130

in vec3 vs_out_col;
in vec3 vs_out_acc;
in float vs_out_mass;


out vec4 fs_out_col;

void main()
{
	float alpha = vs_out_mass;
	vec3 color = vec3(1, 0.4, 0);
	color = mix(color, vec3(1), length(vs_out_acc) / 30);
	fs_out_col = vec4(color,alpha);
}