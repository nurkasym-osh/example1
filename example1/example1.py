import numpy as np
from scipy.sparse import lil_matrix,csc_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

def sdivide(a,b,default=0.0):    
    try:
        if abs(b) < 1e-14:
            return default
        result=a/b
        if np.isinf(result) or np.isnan(result):
            return default
        if abs(result) > 1e10:
            return np.sign(result) * 1e10
        return result
    except:
        return default

def spseudoparabolic(Nx,Ny,max_iter=10000,tol=1e-6):    
    hx=2.0/Nx
    hy=2.0/Ny
    x=np.linspace(-1,1,Nx+1)
    y=np.linspace(-1,1,Ny+1)
    X,Y=np.meshgrid(x,y)
    
    # Initial approximation
    u=np.zeros((Ny+1,Nx+1))
    
    # Boundary conditions
    for j in range(Ny+1):
        if y[j]>=0:
            u[j,0]=np.sin(np.pi*y[j])
            u[j,-1]=0
    
    for i in range(Nx+1):
        if x[i]<=0:
            u[0,i]=1-x[i]**2
        if -1<=x[i]<=0:
            u[-1,i]=1-x[i]**2
    
    # Relaxation parameters
    base_tau=0.001           

    for iter in range(max_iter):
        u_old=u.copy()
        max_change=0.0
        
        # Solution in the D1 domain
        for i in range(2,Nx-2):
            for j in range(1,Ny-1):
                if x[i] > 0 and y[j] > 0:
                    nvalume=upD1(u_old,i,j,hx,hy,x[i],y[j])
                    max_change=max(max_change,abs(nvalume-u[j,i]))
                    u[j,i]=nvalume
                    
        # Solution in the D2 domain
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                if x[i] > 0 and y[j] < 0:
                    nvalume=upD2(u_old,i,j,hx,hy,x[i],y[j])
                    max_change=max(max_change,abs(nvalume-u[j,i]))
                    u[j,i]=nvalume
                    
        # Solution in the D3 domain
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                if x[i] < 0 and y[j] > 0:
                    nvalume=upD3(u_old,i,j,hx,hy,x[i],y[j])
                    max_change=max(max_change,abs(nvalume-u[j,i]))
                    u[j,i]=nvalume
        
        # Convergence check
        if max_change < tol:
            print(f"Convergence was achieved at the iteration"\
                  "{iter} with an error {max_change:.2e}")
            break
            
        # Smoothing the solution every 100 iterations
        if iter % 100 == 0:
            u=0.9 * u+0.1 * u_old
    return u,x,y

def upD1(u,i,j,hx,hy,x,y):
    if i+2>=u.shape[1]:
        return u[j,i]
        
    # Calculation of derivatives
    uxxx=sdivide(u[j,i+2]-3*u[j,i+1]+3*u[j,i]-u[j,i-1],hx**3)
    uxy=sdivide(u[j+1,i+1]-u[j+1,i-1]-u[j-1,i+1]+u[j-1,i-1],4*hx*hy)
    ux=sdivide(u[j,i+1]-u[j,i-1],2*hx)
    
    # Relaxation parameter with constraint
    tau=min(0.001,1.0/(1+abs(uxxx)+abs(uxy)+abs(ux)))
    
    # Calculating a new value with checking
    delta=tau*(uxxx-uxy+ux+u[j,i])
    if np.isnan(delta) or np.isinf(delta):
        return u[j,i]
    
    nvalume=u[j,i]-delta
    if np.isnan(nvalume) or np.isinf(nvalume):
        return u[j,i]
        
    return nvalume

def upD2(u,i,j,hx,hy,x,y):
    if i<=0 or i>=u.shape[1]-1 or j<=0 or j>=u.shape[0]-1:
        return u[j,i]
        
    # Calculation of derivatives
    uxxy = sdivide(
        (
            u[j+1,i+1]-2*u[j,i+1]+u[j-1,i+1]
            - u[j+1,i-1]+2*u[j,i-1]-u[j-1,i-1]
            ),
        2*hx**2*hy
        )
    uxx=sdivide(u[j,i+1]-2*u[j,i]+u[j,i-1],hx**2)
    uxy = sdivide(
        (
            u[j+1,i+1]-u[j+1,i-1]
            - u[j-1,i+1]+u[j-1,i-1]
            ),
        4*hx*hy
        )
    ux=sdivide(u[j,i+1]-u[j,i-1],2*hx)
    uy=sdivide(u[j+1,i]-u[j-1,i],2*hy)
    
    # Coefficients
    a2=1+y**2
    b2=x*y
    
    # Relaxation parameter with constraint
    tau=min(
        0.001,
        1.0/(
            1+max(abs(a2),abs(b2))+abs(uxxy)+abs(uxx)+abs(uxy)
            )
        )
    
    # Calculating a new value with checking
    delta=tau*(uxxy+a2*uxx+b2*uxy+ux+uy+u[j,i])
    if np.isnan(delta) or np.isinf(delta):
        return u[j,i]
    
    nvalume=u[j,i]-delta
    if np.isnan(nvalume) or np.isinf(nvalume):
        return u[j,i]
        
    return nvalume

def upD3(u,i,j,hx,hy,x,y):
    if i<=0 or i>=u.shape[1]-1 or j<=0 or j>=u.shape[0]-1:
        return u[j,i]
        
    # Calculation of derivatives
    uxyy=sdivide(
        (
            u[j+1,i+1]-2*u[j+1,i]+u[j+1,i-1]
            -u[j-1,i+1]+2*u[j-1,i]-u[j-1,i-1]
            ),
        2*hx*hy**2
        )
    uxy=sdivide(u[j+1,i+1]-u[j+1,i-1]-u[j-1,i+1]+u[j-1,i-1],4*hx*hy)
    uyy=sdivide(u[j+1,i]-2*u[j,i]+u[j-1,i],hy**2)
    ux=sdivide(u[j,i+1]-u[j,i-1],2*hx)
    uy=sdivide(u[j+1,i]-u[j-1,i],2*hy)
    
    # Coefficients
    a3=x
    b3=1+y**2
    
    # Relaxation parameter with constraint
    tau=min(
        0.001,
        1.0/(
            1+max(abs(a3),abs(b3))+abs(uxyy)+abs(uyy)+abs(uxy)
            )
        )
    
    # Calculating a new value with checking
    delta=tau*(uxyy+a3*uxy+b3*uyy+ux+uy+u[j,i])
    if np.isnan(delta) or np.isinf(delta):
        return u[j,i]
    
    nvalume=u[j,i]-delta
    if np.isnan(nvalume) or np.isinf(nvalume):
        return u[j,i]
        
    return nvalume

def visualres(u,x,y):    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D    
    
    fig=plt.figure(figsize=(18,6))
    
    # Fig 1.1
    ax1=fig.add_subplot(131,projection='3d')
    X,Y=np.meshgrid(x,y)
    surf=ax1.plot_surface(X,Y,u,cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    plt.colorbar(surf,ax=ax1)
    ax1.set_title('Solution u(x,y)')
    
    # Fig 1.2
    ax2=fig.add_subplot(132)
    contour=ax2.contour(X,Y,u,levels=15,cmap='viridis')
    plt.colorbar(contour,ax=ax2)    
    
    ax2.axhline(y=0,color='r',linestyle='--',label='Line y=0')
    ax2.axvline(x=0,color='r',linestyle='--',label='Line x=0')    
    
    ax2.text(0.5,0.5,'D₁',horizontalalignment='center')
    ax2.text(0.5,-0.5,'D₂',horizontalalignment='center')
    ax2.text(-0.5,0.5,'D₃',horizontalalignment='center')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Lines and Regions')
    ax2.legend()
    
    # Fig 1.3
    ax3=fig.add_subplot(133)
        
    y_idx=np.argmin(np.abs(y-0.5))
    ax3.plot(x,u[y_idx,:],'b-',label='y=0.5')
        
    x_idx=np.argmin(np.abs(x-0.5))
    ax3.plot(y,u[:,x_idx],'r--',label='x=0.5')
    
    ax3.set_xlabel('Coordinate')
    ax3.set_ylabel('u')
    ax3.set_title('Cross-Section of the Solution')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()    
    
    fig2,ax=plt.subplots(figsize=(8,6))    
    
    residuals=np.zeros_like(u)
    hx=x[1]-x[0]
    hy=y[1]-y[0]
    
    for i in range(2,len(x)-2):
        for j in range(2,len(y)-2):
            if x[i] > 0 and y[j] > 0:  # Область D₁
                residuals[j,i]=calcresiD1(u,i,j,hx,hy)
            elif x[i] > 0 and y[j] < 0:  # Область D₂
                residuals[j,i]=calcresiD2(u,i,j,hx,hy)
            elif x[i] < 0 and y[j] > 0:  # Область D₃
                residuals[j,i]=calcresiD3(u,i,j,hx,hy)
        
    im=ax.imshow(residuals,extent=[x[0],x[-1],y[0],y[-1]],
                   origin='lower',cmap='hot',aspect='auto')
    plt.colorbar(im,ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Distribution of Residuals')
    plt.show()

def calcresiD1(u,i,j,hx,hy):    
    uxxx=(u[j,i+2]-3*u[j,i+1]+3*u[j,i]-u[j,i-1])/(hx**3)
    uxy=(u[j+1,i+1]-u[j+1,i-1]-u[j-1,i+1]+u[j-1,i-1])/(4*hx*hy)
    ux=(u[j,i+1]-u[j,i-1])/(2*hx)
    return abs(uxxx-uxy+ux+u[j,i])

def calcresiD2(u,i,j,hx,hy):    
    uxxy=(
        u[j+1,i+1]-2*u[j,i+1]+u[j-1,i+1]
        -u[j+1,i-1]+2*u[j,i-1]-u[j-1,i-1]
        )/(2*hx**2*hy)

    uxx=(u[j,i+1]-2*u[j,i]+u[j,i-1])/(hx**2)
    uxy=(u[j+1,i+1]-u[j+1,i-1]-u[j-1,i+1]+u[j-1,i-1])/(4*hx*hy)
    ux=(u[j,i+1]-u[j,i-1])/(2*hx)
    uy=(u[j+1,i]-u[j-1,i])/(2*hy)
    return abs(uxxy+uxx+uxy+ux+uy+u[j,i])

def calcresiD3(u,i,j,hx,hy):    
    uxyy=(
        u[j+1,i+1]-2*u[j+1,i]+u[j+1,i-1]
        -u[j-1,i+1]+2*u[j-1,i]-u[j-1,i-1]
        )/(2*hx*hy**2)
    uxy=(u[j+1,i+1]-u[j+1,i-1]-u[j-1,i+1]+u[j-1,i-1])/(4*hx*hy)
    uyy=(u[j+1,i]-2*u[j,i]+u[j-1,i])/(hy**2)
    ux=(u[j,i+1]-u[j,i-1])/(2*hx)
    uy=(u[j+1,i]-u[j-1,i])/(2*hy)
    return abs(uxyy+uxy+uyy+ux+uy+u[j,i])

def main():    
    Nx=Ny=40    
    print("Start of Computations...")
    u,x,y=spseudoparabolic(Nx,Ny)
    print("Computations Completed")    
    visualres(u,x,y)

if __name__ == "__main__":
    main()