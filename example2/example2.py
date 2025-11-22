import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

def sdivide(a, b, default=0.0):    
    try:
        if abs(b) < 1e-14:
            return default
        result = a/b
        if np.isinf(result) or np.isnan(result):
            return default
        if abs(result) > 1e10:
            return np.sign(result) * 1e10
        return result
    except:
        return default

def spseudoparabolic2(Nx, Ny, max_iter=10000, tol=1e-6):    
    hx = 2.0/Nx
    hy = 2.0/Ny
    x = np.linspace(-1, 1, Nx+1)
    y = np.linspace(-1, 1, Ny+1)
    X, Y = np.meshgrid(x, y)
    u = np.zeros((Ny+1, Nx+1))  
    for j in range(Ny+1):
        if y[j] >= 0:
            u[j,0] = np.exp(-y[j]**2)
            u[j,-1] = 0
    
    for i in range(Nx+1):
        if x[i] <= 0:
            u[0,i] = np.cos(np.pi*x[i])
        if -1 <= x[i] <= 0:
            u[-1,i] = np.exp(-x[i]**2)            
    
    for iter in range(max_iter):
        u_old = u.copy()
        max_change = 0.0       
        for i in range(2, Nx-2):
            for j in range(1, Ny-1):
                if x[i] > 0 and y[j] > 0:
                    new_val = updateD1(u_old, i, j, hx, hy, x[i], y[j])
                    max_change = max(max_change, abs(new_val - u[j,i]))
                    u[j,i] = new_val                    

        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                if x[i] > 0 and y[j] < 0:
                    new_val = updateD2(u_old, i, j, hx, hy, x[i], y[j])
                    max_change = max(max_change, abs(new_val - u[j,i]))
                    u[j,i] = new_val                    

        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                if x[i] < 0 and y[j] > 0:
                    new_val = updateD3(u_old, i, j, hx, hy, x[i], y[j])
                    max_change = max(max_change, abs(new_val - u[j,i]))
                    u[j,i] = new_val
        
        if max_change < tol:
            print(f"Сходимость достигнута на итерации {iter} с ошибкой {max_change:.2e}")
            break
            
        if iter % 100 == 0:
            u = 0.9 * u + 0.1 * u_old
            
    return u, x, y

def updateD1(u, i, j, hx, hy, x, y):
    if i+2 >= u.shape[1]:
        return u[j,i]

    uxxx = sdivide(u[j,i+2] - 3*u[j,i+1] + 3*u[j,i] - u[j,i-1], hx**3)
    uxy = sdivide(u[j+1,i+1] - u[j+1,i-1] - u[j-1,i+1] + u[j-1,i-1], 4*hx*hy)
    ux = sdivide(u[j,i+1] - u[j,i-1], 2*hx)    

    a1 = 1 + x**2
    d1 = y
    
    tau = min(0.001, 1.0/(1 + abs(uxxx) + abs(uxy) + abs(ux)))
    delta = tau*(uxxx - uxy + a1*ux + d1*u[j,i])
    
    if np.isnan(delta) or np.isinf(delta):
        return u[j,i]
    
    new_val = u[j,i] - delta
    if np.isnan(new_val) or np.isinf(new_val):
        return u[j,i]
        
    return new_val

def updateD2(u, i, j, hx, hy, x, y):
    if i <= 0 or i >= u.shape[1]-1 or j <= 0 or j >= u.shape[0]-1:
        return u[j,i]        

    uxxy = sdivide(u[j+1,i+1] - 2*u[j,i+1] + u[j-1,i+1] - u[j+1,i-1] + 2*u[j,i-1] - u[j-1,i-1], 2*hx**2*hy)
    uxx = sdivide(u[j,i+1] - 2*u[j,i] + u[j,i-1], hx**2)
    uxy = sdivide(u[j+1,i+1] - u[j+1,i-1] - u[j-1,i+1] + u[j-1,i-1], 4*hx*hy)
    ux = sdivide(u[j,i+1] - u[j,i-1], 2*hx)
    uy = sdivide(u[j+1,i] - u[j-1,i], 2*hy)    

    a2 = 1 + y**2
    b2 = x*y
    
    tau = min(0.001, 1.0/(1 + max(abs(a2), abs(b2)) + abs(uxxy) + abs(uxx) + abs(uxy)))
    delta = tau*(uxxy + a2*uxx + b2*uxy + ux + uy + u[j,i])
    
    if np.isnan(delta) or np.isinf(delta):
        return u[j,i]
    
    new_val = u[j,i] - delta
    if np.isnan(new_val) or np.isinf(new_val):
        return u[j,i]
        
    return new_val

def updateD3(u, i, j, hx, hy, x, y):
    if i <= 0 or i >= u.shape[1]-1 or j <= 0 or j >= u.shape[0]-1:
        return u[j,i]        

    uxyy = sdivide(u[j+1,i+1] - 2*u[j+1,i] + u[j+1,i-1] - u[j-1,i+1] + 2*u[j-1,i] - u[j-1,i-1], 2*hx*hy**2)
    uxy = sdivide(u[j+1,i+1] - u[j+1,i-1] - u[j-1,i+1] + u[j-1,i-1], 4*hx*hy)
    uyy = sdivide(u[j+1,i] - 2*u[j,i] + u[j-1,i], hy**2)
    ux = sdivide(u[j,i+1] - u[j,i-1], 2*hx)
    uy = sdivide(u[j+1,i] - u[j-1,i], 2*hy)    

    a3 = x
    b3 = 1 + y**2
    
    tau = min(0.001, 1.0/(1 + max(abs(a3), abs(b3)) + abs(uxyy) + abs(uyy) + abs(uxy)))
    delta = tau*(uxyy + a3*uxy + b3*uyy + ux + uy + u[j,i])
    
    if np.isnan(delta) or np.isinf(delta):
        return u[j,i]
    
    new_val = u[j,i] - delta
    if np.isnan(new_val) or np.isinf(new_val):
        return u[j,i]
        
    return new_val

def visualizeres(u, x, y):    
    fig = plt.figure(figsize=(18, 6))   

    ax1 = fig.add_subplot(131, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax1.plot_surface(X, Y, u, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    plt.colorbar(surf, ax=ax1)
    ax1.set_title('Solution u(x,y)')   

    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, u, levels=15, cmap='viridis')
    plt.colorbar(contour, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--', label='y=0')
    ax2.axvline(x=0, color='r', linestyle='--', label='x=0')
    ax2.text(0.5, 0.5, 'D₁', horizontalalignment='center')
    ax2.text(0.5, -0.5, 'D₂', horizontalalignment='center')
    ax2.text(-0.5, 0.5, 'D₃', horizontalalignment='center')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Lines and Regions')
    ax2.legend()
    
    ax3 = fig.add_subplot(133)
    y_idx = np.argmin(np.abs(y - 0.5))
    ax3.plot(x, u[y_idx,:], 'b-', label='y=0.5')
    x_idx = np.argmin(np.abs(x - 0.5))
    ax3.plot(y, u[:,x_idx], 'r--', label='x=0.5')
    ax3.set_xlabel('Coordinate')
    ax3.set_ylabel('u')
    ax3.set_title('Cross-Section of the Solution')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def main_example2():    
    Nx = Ny = 40    
    print("Start of Computations for the Example 2...")
    u, x, y = spseudoparabolic2(Nx, Ny)
    print("Computations Completed")        
    visualizeres(u, x, y)

if __name__ == "__main__":
    main_example2()