import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Net(nn.Module):
    def __init__(self, hidden_dim=50, num_layers = 4):
        super(Net, self).__init__()
        
        layers = []
        # Input layer: 2D coordinates (x, y)
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer: solution u(x,y)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    

class DeepRitzSolver:
    def __init__(self, net, net_old, Para):
        self.net = net.to(device)
        self.net_old = net_old.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=Para.lr)
        self.N = Para.N
        self.T = Para.T
        self.tau = Para.T / Para.N
        
    def energy_loss(self, interior_points, t, Para, boundary_points=None, initial = False):
        """
        Compute the Ritz energy functional for Poisson equation:
        E(u) = 0.5 * ∫|∇u|² dx - ∫f u dx
        """
        # Interior points
        x_int = interior_points.clone().requires_grad_(True)
        #uaux = x_int[:,0:1] * (1 - x_int[:,0:1]) * x_int[:,1:2] * (1 - x_int[:,1:2])
        u_int = self.net(x_int) #* uaux
        u_int_old = self.net_old(x_int)
        
        # Compute gradient ∇u
        grad_u = grad(u_int, x_int, grad_outputs=torch.ones_like(u_int),
                     create_graph=True)[0]
        
        # Compute |∇u|²
        grad_sq = torch.sum(grad_u**2, dim=1, keepdim=True)
        
        # Source term f(x,y)
        f = self.source_term(x_int, t, Para)
        
        # Energy functional
        if initial == True:
            uinit = self.exact_solution(x_int, t=0, Para=Para)
            energy = u_int**2/self.tau + 0.5 * grad_sq - (f + uinit/self.tau) * u_int
        else:
            energy = u_int**2/self.tau + 0.5 * grad_sq - (f + u_int_old/self.tau) * u_int
        
        # Boundary loss (if boundary points provided)
        boundary_loss = 0.0
        if boundary_points is not None:
            x_bnd = boundary_points.clone().requires_grad_(True)
            u_bnd = self.net(x_bnd)
            u_exact_bnd = self.exact_solution(x_bnd, t, Para)
            #print(u_exact_bnd)
            boundary_loss = 200*torch.mean((u_bnd - u_exact_bnd)**2)

        
        return torch.mean(energy) + boundary_loss
    
    def source_term(self, x, t, Para):
        """Source term f(x,y) = 2π² sin(πx) sin(πy)"""
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        return 2 * Para.k1**2 * np.pi**2 * t**2 * torch.sin(Para.k1 * np.pi * x_coord) * torch.sin(Para.k1 * np.pi * y_coord) + 2 * t * torch.sin(Para.k1 * np.pi * x_coord) * torch.sin(Para.k1 * np.pi * y_coord)
    
    def exact_solution(self, x, t, Para):
        """Exact solution u(x,y) = sin(πx) sin(πy)"""
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        return t**2 * torch.sin(Para.k1 * np.pi * x_coord) * torch.sin(Para.k1 * np.pi * y_coord) 
    
    def train(self, num_epochs, interior_points, Para, boundary_points=None):
        losses = []
        t = 0
        for i in range(0, self.N):
            if i == 0:
                initi = True
            else:
                initi = False
            print("Step ", i)
            t += self.tau
            print(" t = ", t)
            for epoch in range(num_epochs):
                self.optimizer.zero_grad()
                loss = self.energy_loss(interior_points, t, Para, boundary_points, initi)
                loss.backward()
                self.optimizer.step()
            
                losses.append(loss.item())
            
                if epoch % 1000 == 0:
                    #losses.append(loss.item())
                    print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

            self.net_old.load_state_dict(self.net.state_dict())
        
        return losses
    
    def predict(self, x):
        """Predict solution at points x"""
        self.net.eval()
        #uaux = x[:,0:1] * (1 - x[:,0:1]) * x[:,1:2] * (1 - x[:,1:2])
        with torch.no_grad():
            return self.net(x) #* uaux
        
def generate_training_data(num_interior=1000, num_boundary=400):
    x1d = torch.linspace(0, 1, num_interior + 1)
    x1d_new = x1d[0:-1] + 1/(2*num_interior)  # Exclude boundaries
    x2d = torch.linspace(0, 1, num_interior + 1)
    x2d_new = x2d[0:-1] + 1/(2*num_interior)  # Exclude boundaries
    X1, X2 = torch.meshgrid(x1d_new, x2d_new, indexing='ij')
    X = torch.cat([X1.reshape(-1,1), X2.reshape(-1,1)], dim=1)

    boundary = []
    # Bottom boundary (y=0)
    x_btm = torch.linspace(0, 1, num_boundary + 1).unsqueeze(1)
    x_btm_new = x_btm[0:-1] + 1/(2*num_boundary)
    y_btm = torch.zeros_like(x_btm_new)
    boundary.append(torch.cat([x_btm_new, y_btm], dim=1))
    # Top boundary (y=1)
    x_top = torch.linspace(0, 1, num_boundary + 1).unsqueeze(1)
    x_top_new = x_top[0:-1] + 1/(2*num_boundary)
    y_top = torch.ones_like(x_top_new)
    boundary.append(torch.cat([x_top_new, y_top], dim=1))
    # Left boundary (x=0)
    y_left = torch.linspace(0, 1, num_boundary + 1).unsqueeze(1)
    y_left_new = y_left[0:-1] + 1/(2*num_boundary)  # Exclude corners
    x_left = torch.zeros_like(y_left_new)
    boundary.append(torch.cat([x_left, y_left_new], dim=1))
    # Right boundary (x=1)
    y_right = torch.linspace(0, 1, num_boundary + 1).unsqueeze(1)
    y_right_new = y_right[0:-1] + 1/(2*num_boundary)  # Exclude corners
    x_right = torch.ones_like(y_right_new)
    boundary.append(torch.cat([x_right, y_right_new], dim=1))

    boundary_points = torch.cat(boundary, dim=0)
    #print(boundary_points.shape)
    return X, boundary_points

class P:
    def __init__(self, hidden_dim=50, num_layers=4, lr=0.001, num_epochs=5000, num_interior=1000, num_boundary=400, N = 100, T = 1, k1=1):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_interior = num_interior
        self.num_boundary = num_boundary
        self.N = N
        self.T = T
        self.k1 = k1

def main(Para):
    # Generate training data
    interior_points, boundary_points = generate_training_data(Para.num_interior, Para.num_boundary)
    
    # Initialize network and solver
    net = Net(hidden_dim=Para.hidden_dim, num_layers=Para.num_layers)
    net1 = Net(hidden_dim=Para.hidden_dim, num_layers=Para.num_layers)
    solver = DeepRitzSolver(net, net1, Para)
    
    # Train the model
    print("Training started...")
    losses = solver.train(Para.num_epochs, interior_points, Para, boundary_points)
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    #plt.yscale('log')
    
    # Evaluate on test grid
    n_test = Para.num_interior
    x = torch.linspace(0, 1, n_test, device=device)
    y = torch.linspace(0, 1, n_test, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    test_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Predict solution
    u_pred = solver.predict(test_points)
    u_exact = solver.exact_solution(test_points, t=Para.T, Para=Para)
    
    # Reshape for plotting
    U_pred = u_pred.reshape(n_test, n_test).cpu().numpy()
    U_exact = u_exact.reshape(n_test, n_test).cpu().numpy()
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # Plot results
    plt.subplot(1, 2, 2)
    error = np.abs(U_pred - U_exact)
    plt.contourf(X_np, Y_np, error, levels=50)
    plt.colorbar()
    plt.title('Absolute Error')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate L2 error
    l2_error = torch.sqrt(torch.mean((u_pred - u_exact)**2))
    print(f"L2 Error: {l2_error.item():.6f}")
    
    # 3D visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Predicted solution
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X_np, Y_np, U_pred, cmap='viridis', alpha=0.8)
    ax1.set_title('Predicted Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    
    # Exact solution
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X_np, Y_np, U_exact, cmap='viridis', alpha=0.8)
    ax2.set_title('Exact Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    
    # Error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X_np, Y_np, error, cmap='hot', alpha=0.8)
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Error')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()