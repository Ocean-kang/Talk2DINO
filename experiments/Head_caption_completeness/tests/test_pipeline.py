import socket
import unittest
from experiments.Head_caption_completeness.pipeline import free_port,paths,workenv
from experiments.Head_caption_completeness.core import BENCHMARKS

class TestLauncher(unittest.TestCase):
    def test_port_is_os_allocated(self):
        p=free_port()
        self.assertTrue(1024<=p<=65535)
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1',p))
    def test_unique_weights_and_eval(self):
        self.assertNotEqual(paths('E3',123)[0],paths('E3',124)[0])
        self.assertNotEqual(paths('E3',123)[0],paths('E4',123)[0])
        self.assertEqual(set(BENCHMARKS),{'voc20','voc','coco_stuff','ade20k','context59'})
    def test_env(self):
        env=workenv('2',{'MASTER_PORT':'31987'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'2')
        self.assertEqual(env['MASTER_PORT'],'31987')
