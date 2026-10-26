"""Train a variational autoencoder on low-dimensional fictional expression profiles and sample new mathematical profiles."""
from kit import *
TITLE='Variational Autoencoder for Synthetic Expression'
LIVE=False

def dataset(seed,n):
    rng=np.random.default_rng(seed);latent=rng.normal(size=(n,3));basis=np.random.default_rng(8).normal(0,.4,(3,20));return latent@basis+rng.normal(0,.1,(n,20))
def init():
    rng=np.random.default_rng(7);return [rng.normal(0,.1,(20,12)),np.zeros(12),rng.normal(0,.1,(12,3)),np.zeros(3),rng.normal(0,.1,(12,3)),np.zeros(3),rng.normal(0,.1,(3,20)),np.zeros(20)]
def forward(p,x,eps):
    h=np.tanh(x@p[0]+p[1]);mu=h@p[2]+p[3];lv=h@p[4]+p[5];std=np.exp(lv/2);z=mu+std*eps;pred=z@p[6]+p[7];return h,mu,lv,std,z,pred
def loss_grad(p,x,eps,beta=.02):
    h,mu,lv,std,z,pred=forward(p,x,eps);n=len(x);loss=np.mean((pred-x)**2)+beta*.5*np.mean(np.sum(mu*mu+np.exp(lv)-1-lv,axis=1));dp=2*(pred-x)/x.size;dz=dp@p[6].T;dm=dz+beta*mu/n;dl=dz*eps*std*.5+beta*.5*(np.exp(lv)-1)/n;dh=(dm@p[2].T+dl@p[4].T)*(1-h*h)
    return float(loss),[x.T@dh,dh.sum(axis=0),h.T@dm,dm.sum(axis=0),h.T@dl,dl.sum(axis=0),z.T@dp,dp.sum(axis=0)]
def train(x,steps=450):
    p=init();rng=np.random.default_rng(9);history=[]
    for _ in range(steps):
        value,g=loss_grad(p,x,rng.normal(size=(len(x),3)));history.append(value)
        for v,grad in zip(p,g):v-=.12*np.clip(grad,-2,2)
    return p,history
def run(d,a):
    x=dataset(a.seed,500);test=dataset(a.seed+1,200);mean,scale=standardize_fit(x);z=(x-mean)/scale;zt=(test-mean)/scale;p,h=train(z);reconstruction=forward(p,zt,np.zeros((len(zt),3)))[-1];generated=np.random.default_rng(5).normal(size=(30,3))@p[6]+p[7]
    return dict(test_reconstruction_mse=float(np.mean((reconstruction-zt)**2)),mean_baseline_mse=float(np.mean(zt**2)),initial_loss=h[0],final_loss=h[-1],latent_dimensions=3,generation_scope='Samples from a toy latent Gaussian; no verified biological meaning'),dict(**{f'p{i}':v for i,v in enumerate(p)},mean=mean,scale=scale,generated_standardized_profiles=generated)

if __name__=="__main__":main(run)
