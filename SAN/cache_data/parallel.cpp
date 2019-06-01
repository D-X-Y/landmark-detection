//g++ -fopenmp parallel.cpp
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[]){
	if( argc != 3 ){
		fprintf(stderr, "usage: %s commands threads\n", argv[0]);
		return -1;
	}
	FILE *in = fopen(argv[1], "r");
	assert(in != NULL);
	vector<string> cmds;
	char buf[10240];
	while(fscanf(in, "%[^\n]\n", buf) > 0 ){
		cmds.push_back(buf);
	}
	fclose(in);
	int threads;
	sscanf(argv[2], "%d", &threads);
	omp_set_num_threads(threads);
	int cmds_count = cmds.size();
	fprintf(stderr, "read %d commands, using %d threads\n", cmds_count, threads);
	#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < cmds_count; i++){
		//string cmd = string("eval ") + cmds[i];
		string cmd = cmds[i];
		system(cmd.c_str());
		if( i % 100 == 0 ){
			fprintf(stderr, "completed %d/%d\n", i, cmds_count);
		}
	}
	return 0;
}
