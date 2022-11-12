In general, it seems that the pure exploration algorithm performs the worst out of the 7. Not only does it have the
highest cumulative regret for this particular run, it also has low variance, which means that it consistently performs
at this level. We also saw in class that the pure greedy algorithm has linear regret. However, in the plots, it seems to
do reasonably well compared to the others. This is because for this particular scenario, there are only 5 different arms,
which means that the greedy algorithm can often find the best arm in a reasonable time. If there were significantly more
arms (with the same time horizon), I would expect this algorithm to perform worse. We also see that the confidence band
is quite large, which is expected for this algorithm. We also see that the initial regret for the ETC algorithm is large,
and then abruptly shrinks: this makes sense because in this scenario, this algorithm chooses the optimal arm quite often
after the exploration stage.

The epsilon-greedy and UCB algorithms performed somewhat similarly to the pure explore algorithm in this scenario. For
the epsilon-greedy approach, this makes sense, as for the given parameters, the value of epsilon was quite large, reaching
its minimum value of about 0.6 only at the end of the algorithm. Therefore, for a large part of the time, this algorithm
would perform like the pure exploration approach, which is not good. From the graphs, it looks like these two algorithms
have a lower variance, but a higher average cumulative regret than some of the other algorithms like Thompson sampling
and the Gittins index.

The Thompson sampling and Gittins index algorithms seemed to perform the best overall. Their 95% cumulative regret is
still close or below some of the others' average regret, and for this particular run, the Gittins index had the lowest
cumulative regret. This makes sense, as we saw in class that these two algorithms were asymptotically optimal. Here,
having T = 100 seemed to be large enough to fix the problem of Thompson sampling not being greedy enough.
