Convergence of SGD

$E[F(w^t)-F(w^*)]\le \frac{log(t)}{\sqrt{t}}$

$\forall x>0, t> 0, P(x>t*E[x]) \le \frac{1}{t}$

$x=F(w^t)-F(w^*)\ge0$

$t=1000$ w.p. $0.001,$

$ F(w^t)-F(w^*)\ge1000*E[x]$

$ \rightarrow$ w.p. $0.999,$

$F(w^t)-F(w^*)\le 1000*E[x] \le 1000*\frac{log(t)}{\sqrt{t}}$