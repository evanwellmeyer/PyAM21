/* macOS stub for FMS affinity.c (Linux sched_getaffinity is unavailable). Not physics. */
int get_cpu_affinity(void) { return 0; }
int get_cpu_affinity_(void) { return get_cpu_affinity(); }
void set_cpu_affinity(int cpu) { (void)cpu; }
void set_cpu_affinity_(int *cpu) { set_cpu_affinity(*cpu); }
