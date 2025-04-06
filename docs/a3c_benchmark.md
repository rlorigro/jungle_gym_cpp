## A3C benchmarking

Benchmarking A3C requires controlling for several factors:
- **number of episodes**: we fix n_episodes at 1000
- **model-related factors**: since the update step occurs after T steps OR upon termination, time until termination by 
the environment depends on the model. To account for this factor, the benchmark is run using both the initial and final 
model, assuming that the trained ("final") model life expectancy is ~T time steps
- **parallel tensor operations**: libtorch's default use of OMP for multithreaded Tensor operations are disabled with
`torch::set_num_threads(1)` to prevent torch threads from being conflated with a3c worker threads

The benchmarking executable loops over n_threads. Steps per episode is 16 (not shown):

```c++
    // --- loop over n_threads ---
    for (auto n: {1,2,4,8,16}) {
        hyperparams.n_threads = n;
        hyperparams.n_episodes = 1000;
        hyperparams.silent = true;

        // --- Initial ---
        // Measure time to train initial model (short episodes, update is bottleneck)
        A3CAgent c(hyperparams, actor, critic);
        c.load(initial_output_dir / "actor.pt", initial_output_dir / "critic.pt");

        auto start_time = std::chrono::high_resolution_clock::now();
        c.train(env);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed_time = end_time - start_time;
        std::cout << "initial\t" << n << "\t" << elapsed_time.count() << "\n";

        // --- Final ---
        // Measure time to train final model (long episodes, update is less of bottleneck)
        A3CAgent d(hyperparams, actor, critic);
        d.load(final_output_dir / "actor.pt", final_output_dir / "critic.pt");

        start_time = std::chrono::high_resolution_clock::now();
        d.train(env);
        end_time = std::chrono::high_resolution_clock::now();

        elapsed_time = end_time - start_time;
        std::cout << "final  \t" << n << "\t" << elapsed_time.count() << "\n";

    }
```

The results show that there is a negligible benefit beyond 8 threads when using the SnakeEnv (time in sec):


| n_threads | initial  | final   |
|-----------|----------|---------|
| 1         | 51.8932  | 91.1588 |
| 2         | 26.867   | 50.4489 |
| 4         | 17.265   | 28.1729 |
| 8         | 13.6072  | 20.7357 |
| 16        | 14.5809  | 19.288  |


However, surprisingly, perf reports very low lock contention when profiling the same benchmark at 16 threads. This
is unlikely to be accurate...
```
sudo perf lock record ./benchmark_a3c
perf lock contention
 contended   total wait     max wait     avg wait         type   caller

     10051    108.29 ms     75.93 us     10.77 us     spinlock   Unknown
      2475     88.99 ms     77.47 us     35.96 us     spinlock   Unknown
      9501     88.67 ms     60.68 us      9.33 us     spinlock   Unknown
        70     73.34 ms      1.78 ms      1.05 ms      rwsem:W   Unknown
        50      2.60 ms     78.34 us     51.91 us     spinlock   Unknown
       109      1.87 ms     37.17 us     17.13 us     spinlock   Unknown
       118      1.57 ms     30.77 us     13.31 us     spinlock   Unknown
        28      1.50 ms     78.92 us     53.59 us     spinlock   Unknown
        27      1.46 ms     78.19 us     53.94 us     spinlock   Unknown
        28      1.39 ms     78.95 us     49.81 us     spinlock   Unknown
        25      1.35 ms     78.31 us     54.17 us     spinlock   Unknown
```

We can check the flame graph to see what is the relative proportion of time spent on rollouts vs optimizer:

![a3c_16_thread_flamegraph.png](../data/a3c_16_thread_flamegraph.png)

It appears that in the A3CAgent::train method, the RMSPropAsync::step method accounts for about 15% of the train time.
It is worth noting that in this implementation the workers are responsible for computing the updates themselves, by
calling the member function of the global RMSPropAsync instance. 

Within the step function, the majority of the cost is in computing the update:

![a3c_16_thread_flamegraph.png](../data/a3c_16_thread_flamegraph_step_fn.png)

Very little time is attributed to the mutex (highlighted), surprisingly. This suggests that some other form of thread 
overhead could be the bottleneck, or that the profiling is not properly capturing the wait time. 

To test if memory-related overhead may be improved with mimalloc (a custom memory allocator) I added the option to 
compile with it. The result is a notable improvement in speed, but not necessarily an improvement in thread-related 
bottlenecks, since the plateau is reached at around the same time:

| n_threads | initial  | final   |
|-----------|----------|---------|
| 1         | 33.4494  | 70.6786 |
| 2         | 17.9071  | 37.8691 |
| 4         | 10.7172  | 22.1412 |
| 8         | 9.1711   | 18.2394 |
| 16        | 9.44162  | 18.6533 |

(time in sec)

This plot summarizes the findings:

![benchmark_a3c_plot.png](../data/benchmark_a3c_plot.png)

### Manually profiling

As it turns out, using a simple timer and an atomic counter gives much more informative results:

| n  | total_s initial | wait_s initial | total_s final | wait_s final |
|----|-----------------|----------------|---------------|--------------|
| 1  | 33.7            | 0.00875        | 71.7          | 0.0109       |
| 2  | 18.1            | 0.405          | 38.3          | 0.257        |
| 4  | 10.7            | 0.673          | 21.7          | 0.432        |
| 8  | 9.34            | 1.31           | 18.1          | 0.742        |
| 16 | 9.26            | 4.16           | 18.9          | 1.72         |

where total_s is the wall clock time and wait_s is the average total mutex wait time per thread (total_wait_time_s/n).

Clearly as more threads are added, the mutex wait time becomes a more significant portion of the total time. Also, we 
now see that wait time is less significant when episodes are more computationally intensive (longer) than the update 
step by looking at the "final" columns.

Overall this suggests that a more fine grained locking or some specialized lock-minimal data structure is needed.

