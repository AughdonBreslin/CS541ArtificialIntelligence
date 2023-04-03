$F(w)$ is smooth if for any $w_1, w_2 \in \mathbb{R}^d$,
$||\nabla F(w_2) - \nabla F(w_1)||_2 \le L||w_2-w_1||_2$.

In other words, the difference of gradients over the difference of weights must be bounded by a positive L.
$\frac{||\nabla F(w_2) - \nabla F(w_1)||_2}{||w_2-w_1||_2} \le L$.



Homework Examples:

$F(w)=e^{-w}, w_1, w_2$

- $\frac{||\nabla F(w_2) - \nabla F(w_1)||_2}{||w_2-w_1||_2} \le L$
- $\frac{||-e^{-w_2}+e^{-w_1}||_2}{||w_2-w_1||_2} \le L$
- Non-constant L -> non-smooth

$F(w)=\frac{1}{1+e^{-w}}$

- $\frac{||\nabla F(w_2) - \nabla F(w_1)||_2}{||w_2-w_1||_2} \le L$
- $\frac{||\frac{e^{-w_2}}{(1+e^{-w_2})^2}-\frac{e^{-w_1}}{(1+e^{-w_1})^2}||_2}{||w_2-w_1||_2} \approx \frac{e^{w_2}-e^{w_1}}{w_2-w_1}$