import unittest
from app import *
class DomainTests(unittest.TestCase):
    def test_reparameterization_gradient(self):
        x=dataset(1,4);eps=np.random.default_rng(2).normal(size=(4,3));p=init();_,g=loss_grad(p,x,eps);old=p[4][0,0];delta=1e-5;p[4][0,0]=old+delta;hi=loss_grad(p,x,eps)[0];p[4][0,0]=old-delta;lo=loss_grad(p,x,eps)[0];self.assertAlmostEqual((hi-lo)/(2*delta),g[4][0,0],places=5)
    def test_training_decreases(self):
        _,h=train(dataset(1,100),120);self.assertLess(np.mean(h[-10:]),np.mean(h[:10]))
    def test_output_shape(self):self.assertEqual(forward(init(),np.ones((4,20)),np.zeros((4,3)))[-1].shape,(4,20))
