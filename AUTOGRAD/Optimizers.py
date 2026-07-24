class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self):
        """Resets the gradients of all parameters to zero."""
        for p in self.parameters:
            p.gradient = 0.0

    def step(self):
        """Updates parameters based on their gradients."""
        raise NotImplementedError("Subclasses must implement the step() method.")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with Momentum and L2 weight decay parameter default to zero.
    """
    def __init__(self, parameters, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay        
        self.v = [0.0] * len(parameters)
            
    def step(self):
        for i, p in enumerate(self.parameters):
            grad = p.gradient
            if self.weight_decay > 0.0:
                grad += self.weight_decay * p.data
                
            if self.momentum > 0.0:
                self.v[i] = self.momentum * self.v[i] + grad
                step_size = self.v[i]
            else:
                step_size = grad
                
            p.data -= self.learning_rate * step_size


class RMSprop(Optimizer):
    """
    RMSprop Optimizer.
    """
    def __init__(self, parameters, learning_rate=0.01, alpha=0.99, epsilon=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        #moving average of squared gradients
        self.v = [0.0] * len(parameters)

    def step(self):
        for i, p in enumerate(self.parameters):
            grad = p.gradient
            if self.weight_decay > 0.0:
                grad += self.weight_decay * p.data

            # moving average of squared gradients
            self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * (grad ** 2)
            
            #learning rate will be down where gradients are steep, up where flat
            p.data -= self.learning_rate * grad / (self.v[i]**0.5 + self.epsilon)


class Adam(Optimizer):
    """
    Standard Adam optimizer.
    """
    def __init__(self, parameters, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)
        self.t = 0 
            
    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            self.m[i] = self.beta_1 * self.m[i] + (1.0 - self.beta_1) * p.gradient
            self.v[i] = self.beta_2 * self.v[i] + (1.0 - self.beta_2) * (p.gradient ** 2)
            
            m_hat = self.m[i] / (1.0 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta_2 ** self.t)
            
            p.data -= self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)


class AdamW(Optimizer):
    """
    AdamW: Adam with Decoupled Weight Decay.
    """
    def __init__(self, parameters, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, weight_decay=0.01):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)
        self.t = 0 
            
    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if self.weight_decay > 0.0:
                p.data -= self.learning_rate * self.weight_decay * p.data
                
            self.m[i] = self.beta_1 * self.m[i] + (1.0 - self.beta_1) * p.gradient
            self.v[i] = self.beta_2 * self.v[i] + (1.0 - self.beta_2) * (p.gradient ** 2)
            
            m_hat = self.m[i] / (1.0 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta_2 ** self.t)
            
            p.data -= self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)