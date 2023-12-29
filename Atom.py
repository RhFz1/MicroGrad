# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:17:39 2023

@author: Inspired from Andrej Karpathy
"""

import math

from NetVisualizer import draw_dot

class Atom():
    def __init__(self, value,_children=(), _op='', label=''):
        self.value = value 
        self.label = label
        self._prev = set(_children)
        self.grad = 0.00000
        self._backward = lambda : None
        self._op = _op
    def __repr__(self):
        return f"Atom(value={self.value})"
    def __add__(self, other):
        other = other if isinstance(other, Atom) else Atom(other)
        out = Atom(self.value + other.value,(self, other),_op= '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Atom) else Atom(other)
        out = Atom(self.value * other.value,(self, other),_op='*')
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        return out
    def __pow__(self ,other):
        assert isinstance(other, (int ,float)), "supports integer and float exponentiation"
        out = Atom(self.value**other,(self,),_op = '^')
        def _backward():
            self.grad += other*(self.value) ** (other - 1) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        x = self.value
        out = Atom(math.exp(x), (self,),'e^x')
        def _backward():
            self.grad += out.value * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        x = self.value
        out = Atom((math.exp(2 * x) - 1)/(math.exp(2 * x) + 1), (self,),_op=  'tanh')
        def _backward():
            self.grad += (1 - (out.value) ** 2) * out.grad
        out._backward = _backward
        return out
    def sigmoid(self):
        out = Atom((1 / (1 + math.exp(-self.value))), (self, ), _op= 'sigmoid')
        def _backward():
            self.grad += (1 - out.value) * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other
    def __neg__(self):
        return self * -1
    def __truediv__(self, other):
        return self * (other**-1)
    def __sub__(self, other):
        return self + (-other)
    def backward(self):
        order = []
        vis = set()
        def dfs(u):
            if (u not in vis):
                vis.add(u)
                for v in u._prev:
                    dfs(v)
                order.append(u)
        dfs(self)        
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
# Example visualization
x = Atom(-1.0, label = 'x'); y = Atom(2.0, label = 'y'); z = Atom(-3.0, label = 'z')
t = ((y ** 2).exp() / y + z).tanh(); t.label = 't'
t.backward()
draw_dot(t)


    
