#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>

#define MAXCOORD 1000000
#define PRANDMAX 1000000000

int a, b;
int arr[55];
void sprand(int);
int lprand(void);

int main(int argc, char **argv)
{
	int graph_number;
	graph_number = 1000;  // number of graph instances to generate
	int id;
	for (id = 1; id <= graph_number; id++) {
		int factor = PRANDMAX / MAXCOORD;
		int N;
		int i;
		int x, y;
		int seed;

		N = 20;  // number of vertices in the graph instance
		seed = id + 10000;  // random seed

		/* initialize random number generator */
		sprand(seed);

		char filename[128], tempfp[128];
		FILE *fp = NULL;
		strcpy(filename, "tsp20_r_test.txt");  // storage path of the graph instances
		fp = fopen(filename, "a+");
		printf("%s", filename);

		for (i = 1; i <= N; i++) {
			char coordinate[128];
			x = lprand() / factor;
			y = lprand() / factor;
			if (i < N)
				sprintf(coordinate,"%f %f ", 1.0 * x / MAXCOORD, 1.0 * y / MAXCOORD);
			else
				sprintf(coordinate,"%f %f\n", 1.0 * x / MAXCOORD, 1.0 * y / MAXCOORD);
			fprintf(fp, coordinate);
			
		}
		fclose(fp);

	}
	return 0;

	
}

void sprand(int seed)
{
	int i, ii;
	int last, next;
	arr[0] = last = seed;
	next = 1;
	for (i = 1; i<55; i++) {
		ii = (21 * i) % 55;
		arr[ii] = next;
		next = last - next;
		if (next < 0) next += PRANDMAX;
		last = arr[ii];
	}
	a = 0;
	b = 24;
	for (i = 0; i<165; i++) last = lprand();
}

int lprand(void)
{
	long t;
	if (a-- == 0) a = 54;
	if (b-- == 0) b = 54;
	t = arr[a] - arr[b];
	if (t<0) t += PRANDMAX;
	arr[a] = t;
	return t;
}
