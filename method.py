from jax import random, grad, vmap, jit
key = random.PRNGKey(0)
keys = [key for key in random.split(key, 10)]
import jax.numpy as jnp
import jax.nn as jnn
import jax
from jax.example_libraries import optimizers
import numpy as np

"""
Tolga Recep Uçar and Halit Tali (Dec 2024)

Providing efficient numerical solutions to differential equations
is of great importance in the absence of an exact solution. Lately,
artificial neural network based machine learning methods have illustrated
promising results in this regard. We propose a low-cost orthogonal network
with fast training by simple gradient descent. The network is basically a
two-layer neural network learning from orthogonal polynomial based representations.
We demonstrate the efficiency and accuracy of our method on several problems and
present comparisons with various other methods. Our proposal essentially stands out
in comparison to high demand solutions.
"""

def legendre(n, x):
  if n == 0:
    return jnp.ones_like(x)
  elif n == 1:
    return x
  else:
    return ((2*n-1)/n) * x * legendre(n-1, x) - ((n-1)/n) * legendre(n-2, x)

def chebyshev(n, x):
  if n == 0:
    return jnp.ones_like(x)
  elif n == 1:
    return x
  else:
    return 2*x*chebyshev(n-1, x) - chebyshev(n-2, x)

x20 = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])

def proposed(w, x):
  w0 = w[0]
  w1 = w[1]
  x = jnp.tanh(x)
  x = jnp.stack([legendre(i, x) for i in range(2, 7)]).T
  x = x.T if x.shape[0]==1 else x
  h0 = jnp.dot(w0, x)
  o = jnp.dot(w1, jnn.sigmoid(h0))
  return o.squeeze()

params_proposed = [random.normal(keys[0], shape=(5, 5)), random.normal(keys[1], shape=(1, 5))]

def chnn(w, x):
  w = w[0]
  x = jnp.stack([chebyshev(i, x) for i in range(5)]).T
  x = x.T if x.shape[0]==1 else x
  z = jnp.tanh(jnp.dot(w, x))
  return z.squeeze()

params_chnn = [random.normal(keys[0], shape=(1, 5))]

model = proposed
p = params_proposed
x = x20
iv = 1.
iv1 = 0.

def t(w, x):
  # return iv+x*model(w,x) # first order approximation
  return iv+iv1*x+(x**2)*model(w,x) # second order approximation

dtdx = grad(t, 1)
dtdx2 = grad(dtdx, 1)
t_vect = vmap(t, (None, 0))
dtdx_vect = vmap(dtdx, (None, 0))
dtdx2_vect = vmap(dtdx2, (None, 0))

def loss(w, x, lam=1, n=2):
  # eqn = dtdx_vect(w,x) - (4*x**3 + -3*x**2 + 2)
  # eqn = dtdx_vect(w,x) - (x+t_vect(w,x))
  # eqn = dtdx_vect(w,x) + 0.2*t_vect(w,x) - (jnp.cos(x)*jnp.exp(-0.2*x)) # y_0 = 0
  # eqn = dtdx_vect(w,x)-2*x-1 # y_0 = 0
  # eqn = dtdx_vect(w,x) - t_vect(w,x)
  # eqn = dtdx_vect(w,x) + (x+((1+3*x**2)/(1+x+x**3)))*t_vect(w,x) - (x**3 + 2*x + (x**2)*((1+3*x**2)/(1+x+x**3)))
  ## lane-emden ##
  # m=0
  # eqn = dtdx2_vect(w,x)+(2/x)*dtdx_vect(w,x)+1
  # m=1
  # eqn = dtdx2_vect(w,x)+(2/x)*dtdx_vect(w,x)+t_vect(w,x)
  # m=5
  # eqn = dtdx2_vect(w,x)+(2/x)*dtdx_vect(w,x)+(t_vect(w,x)**5)
  # example 2
  # eqn = dtdx2_vect(w,x)+(2/x)*dtdx_vect(w,x)+jnp.exp(t_vect(w,x))
  # example 3
  # eqn = dtdx2_vect(w,x)+(2/x)*dtdx_vect(w,x)+jnp.sinh(t_vect(w,x))
  # example 5
  # eqn = dtdx2_vect(w,x)+(2/x)*dtdx_vect(w,x)+4*(2*jnp.exp(t_vect(w,x))+jnp.exp(t_vect(w,x)/2))
  # example 6
  eqn = dtdx2_vect(w,x)+(2/x)*dtdx_vect(w,x)-2*(2*(x**2)+3)*t_vect(w,x)
  ## abel ##
  # eqn = t_vect(w,x)*dtdx_vect(w,x) + x*t_vect(w,x) + t_vect(w,x)**2 + (x**2)*t_vect(w,x)**3 - (x*jnp.exp(-x) + (x**2)*jnp.exp(-3*x))
  # eqn = dtdx_vect(w,x) - (4*t_vect(w,x) - t_vect(w,x)**3)
  # eqn = dtdx_vect(w,x) - (t_vect(w,x)-2*t_vect(w,x)**2)
  ## COMPARE W/ LS-SVM
  # 5.7
  # eqn = dtdx2_vect(w,x)+(1/x)*dtdx_vect(w,x)-(1/x)*jnp.cos(x)
  # 5.6
  # eqn = dtdx2_vect(w,x)+(1/5)*dtdx_vect(w,x)+t_vect(w,x)-((-1/5)*jnp.exp(-x/5)*jnp.cos(x))
  # 5.5
  # eqn = dtdx2_vect(w,x)+t_vect(w,x)-(2+2*jnp.sin(4*x)*jnp.cos(3*x))
  # 5.4
  # eqn = dtdx_vect(w,x) + (x+((1+3*(x**2))/(1+x+(x**3))))*t_vect(w,x) - ((x**3)+2*x+(x**2)*((1+3*(x**2))/(1+x+(x**3))))
  # eqn = dtdx_vect(w,x)+2*t_vect(w,x)-jnp.sin(x) # iv=1. t in [0,10]
  # eqn = dtdx_vect(w,x)+2*t_vect(w,x)-((x**3)*jnp.sin(x/2))
  # eqn = dtdx_vect(w,x)-((t_vect(w,x)**2)+(x**2))
  # eqn = t_vect(w, x) - jnp.sin(x)
  return (lam/n)*jnp.sum(eqn**2)

grad_loss = jit(grad(loss, 0))

# exact = lambda x: x*(x**3 - x**2 + 2)
# exact = lambda x: -x+2*jnp.exp(x)-1
# exact = lambda x: jnp.exp(-0.2*x)*jnp.sin(x)
# exact = lambda x: x*(x+1)
# exact = lambda x: jnp.exp(x)
# exact = lambda x: x**2 + ((jnp.exp((-x**2)/2))/(x**3 + x + 1))
## lane-emden ##
# m=0
# exact = lambda x: 1.-(x**2/6)
# m=5
# exact = lambda x: 1/jnp.sqrt(1.+(x**2/3))
# example 2 - series expansion
# exact = lambda x: (-1/6)*(x**2)+(1/120)*(x**4)-(8/(21*720))*(x**6)+(122/(81*40320))*(x**8)-(61*67/(495*3628800))*(x**10)
# example 3 - series expansion
# e = jnp.exp(1)
# exact = lambda x: -((e**2 - 1) * x**2) / (12 * e + 1/480 * (e**4 - 1) * x**4 / (12 * e + 1/30240 * (2*e**6 + 3*e**2 - 3*e**4 - 2) * x**6 / (e**3 + 1/26127360 * (61*e**8 - 104*e**6 + 104*e**2 - 61) * x**8 / e**4)))
# example 5
# exact = lambda x: -2*jnp.log(1+(x**2))
# example 6
exact = lambda x: jnp.exp(x**2)

## abel ##
# exact = lambda x: jnp.exp(-x)
# exact = lambda x: 2*jnp.exp(4*x)/jnp.sqrt(jnp.exp(8*x)+15)
# exact = lambda x: 1./(2-jnp.exp(-x))

## COMPARE W/ LS-SVM ##
# 5.6
# exact = lambda x: jnp.exp(-x/5)*jnp.sin(x)
# 5.4
# exact = lambda x: (jnp.exp((-x**2)/2)/(1+x+x**3))+(x**2)
# exact = lambda x: (x**2)+(jnp.exp((-x**2)/2)/(1+x+(x**3))) # 5.4.
# exact = lambda x: jnp.sin(x)/x # 5.7. first derivative of the exact solution
# exact = lambda x: jnp.exp(-x/5)*jnp.sin(x)
# exact = lambda x: (6*jnp.exp(-2*x)+2*jnp.sin(x)-jnp.cos(x))/5 # 5.1.

lr = 1e-3
epoch = 0
grad_normi = 0
while True:
  grads = grad_loss(p, x)
  grad_norm = sum([jnp.linalg.norm(g) for g in grads])
  grad_normf = grad_norm
  if epoch % 1 == 0:
    print('epoch: %.3d loss: %.10f grad_norm: ' % (epoch, loss(p, x)))
  if jnp.abs(grad_normi - grad_normf) / grad_normi < 10**-6:
    break
  for i in range(len(p)):
    p[i] -= lr*grads[i]
  grad_normi = grad_normf
  epoch += 1

def measures(test_points):
  errors = jnp.array([t(p, xi) - exact(xi) for xi in test_points])
  print('MSE: ', jnp.mean(errors**2))
  print('L1: ', jnp.sum(jnp.abs(errors)))
  print('L2: ', jnp.sqrt(jnp.sum(jnp.abs(errors)**2)))
  print('L_infinity: ', jnp.max(jnp.abs(errors)))
  print('MAE: ', jnp.mean(jnp.abs(errors)))

def get_euler(xi, h=.05, talk=False):
  # f = lambda x,y: 4*x**3  -3*x**2 + 2 # y_0 = 0
  f = lambda x,y: y-2*(y**2)
  # f = lambda x,y: x+y
  # f = lambda x,y: -0.2*y+(jnp.cos(x)*jnp.exp(-0.2*x)) # y_0 = 0
  # f = lambda x,y: 2*x+1 # y_0 = 0
  # f = lambda x,y: y
  # f = lambda x,y: -(x+((1+3*x**2)/(1+x+x**3)))*y+(x**3 + 2*x + (x**2)*((1+3*x**2)/(1+x+x**3)))
  xn, y = 0, 1
  n = (int)((xi-0)/h)
  for i in range(1, n+1):
    if talk:
      print('Iteration ', i)
      print('x_n, y_n=', xn, ', ', y)
    y = y + f(xn, y)*h
    xn += h
    if talk:
      print('y= ', y)
  if talk:
    print('Euler approximation for x=', x, ' is ', y)
  return y

def get_rk(xi, h=.05, talk=False):
  # f = lambda x,y: 4*(x**3)-3*(x**2)+2 # y_0 = 0
  # f = lambda x,y: y-2*y**2
  # f = lambda x,y: x+y
  # f = lambda x,y: -0.2*y+(jnp.cos(x)*jnp.exp(-0.2*x)) # y_0 = 0
  # f = lambda x,y: 2*x+1 # y_0 = 0
  # f = lambda x,y: y
  # f = lambda x,y: -(x+((1+3*x**2)/(1+x+x**3)))*y+(x**3 + 2*x + (x**2)*((1+3*x**2)/(1+x+x**3)))
  # f = lambda x,y: 4*y - y**3
  f = lambda x,y: (x*jnp.exp(-x) + (x**2)*jnp.exp(-3*x) - x*y - (y**2) - (x**2)*(y**3))/y
  xn, y = 0, 1.
  n = (int)((xi-xn)/h)
  for _ in range(1,n+1):
    k1 = h*f(xn, y)
    k2 = h*f(xn+h*.5, y+k1*.5)
    k3 = h*f(xn+h*.5, y+k2*.5)
    k4 = h*f(xn+h, y+k3)
    y += (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
    if talk:
      print('k1: ', k1)
      print('k2: ', k2)
      print('k3: ', k3)
      print('k4: ', k4)
      print('y: ', y)
    xn += h
  if talk:
    print('Runge-Kutta approximation for x=', x, ' is ', y)
  return y