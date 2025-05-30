import unittest
import LyndonWords
import importlib
import numpy
importlib.reload(LyndonWords)
from LyndonWords import *

def toSet(lW:rootSystem):
    resultSet = set()
    for i in lW.rootToWordDictionary.values():
        for j in i:
            resultSet.add(str(j))
    return resultSet

class TestStringMethods(unittest.TestCase):
    def test_C_Affine_1(self):
        arr = [4,0,1,2,3]
        expected = set(
            [
                '0','1','2','3','4',
                '4,3','1,2','2,3','0,1','4,3,2','1,2,3','4,3,3','0,1,2','0,1,1','4,3,2,1',
                '4,3,3,2','0,1,2,3','0,1,2,1','4,3,3,2,1','4,3,3,2,2','0,1,2,3,1','4,3,2,1,0',
                '0,1,2,1,2','4,3,3,2,2,1','0,1,2,3,1,2','4,3,3,2,1,0','4,3,2,1,0,1','4,3,3,2,2,1,1','4,3,3,2,2,1,0',
                '4,3,3,2,1,0,1','4,3,2,1,0,1,2','0,1,2,3,1,2,3','4,3,3,2,2,1,1,0','4,3,3,2,2,1,0,1','4,3,3,2,1,0,1,2',
                '4,3,2,1,0,1,2,3','4,3,3,2,2,1,1,0,1','4,3,3,2,2,1,0,1,2','4,3,3,2,1,0,1,2,3','4,3,2,1,4,3,2,1,0','4,3,3,2,2,1,1,0,0',
                '4,3,2,1,0,1,2,4,3,3','4,3,3,2,2,1,1,0,1,2','4,3,3,2,2,1,0,1,2,3','4,3,3,2,2,1,1,0,1,0','4,3,2,1,0,1,2,4,3,3,2','4,3,3,2,2,1,1,0,1,2,3',
                '4,3,3,2,1,4,3,3,2,1,0','4,3,3,2,2,1,1,0,1,2,0','4,3,3,2,2,1,1,0,1,0,1','4,3,2,1,0,1,2,4,3,3,2,1','4,3,3,2,1,0,1,4,3,3,2,2','4,3,3,2,2,1,1,0,1,2,3,0',
                '4,3,3,2,2,1,1,0,1,2,0,1','4,3,3,2,1,0,1,4,3,3,2,2,1','4,3,3,2,2,1,4,3,3,2,2,1,0','4,3,3,2,2,1,1,0,1,2,3,0,1','4,3,2,1,0,1,2,4,3,3,2,1,0','4,3,3,2,2,1,1,0,1,2,0,1,2',
                '4,3,3,2,2,1,0,4,3,3,2,2,1,1','4,3,3,2,2,1,1,0,1,2,3,0,1,2','4,3,3,2,1,0,1,4,3,3,2,2,1,0','4,3,2,1,0,1,2,4,3,3,2,1,0,1','4,3,3,2,2,1,1,4,3,3,2,2,1,1,0','4,3,3,2,2,1,0,4,3,3,2,2,1,1,0',
                '4,3,3,2,1,0,1,4,3,3,2,2,1,0,1','4,3,2,1,0,1,2,4,3,3,2,1,0,1,2','4,3,3,2,2,1,1,0,1,2,3,0,1,2,3','4,3,3,2,2,1,1,4,3,3,2,2,1,1,0,0','4,3,3,2,2,1,0,4,3,3,2,2,1,1,0,1','4,3,3,2,1,0,1,4,3,3,2,2,1,0,1,2',
                '4,3,2,1,0,1,2,4,3,3,2,1,0,1,2,3','4,3,3,2,2,1,1,0,4,3,3,2,2,1,1,0,1','4,3,3,2,2,1,0,1,4,3,3,2,2,1,0,1,2','4,3,3,2,1,0,1,2,4,3,3,2,1,0,1,2,3','4,3,2,1,0,1,2,4,3,2,1,0,1,2,4,3,3','4,3,3,2,2,1,1,0,4,3,3,2,2,1,1,0,0',
                '4,3,2,1,0,1,2,4,3,3,2,1,0,1,2,4,3,3','4,3,3,2,2,1,1,0,4,3,3,2,2,1,1,0,1,2','4,3,3,2,2,1,0,1,4,3,3,2,2,1,0,1,2,3','4,3,3,2,2,1,1,0,0,4,3,3,2,2,1,1,0,1','4,3,2,1,0,1,2,4,3,3,2,1,0,1,2,4,3,3,2',
                '4,3,3,2,2,1,1,0,4,3,3,2,2,1,1,0,1,2,3','4,3,3,2,1,0,1,4,3,3,2,1,0,1,4,3,3,2,2','4,3,3,2,2,1,1,0,0,4,3,3,2,2,1,1,0,1,2','4,3,3,2,2,1,1,0,1,4,3,3,2,2,1,1,0,1,0','4,3,2,1,0,1,2,4,3,3,2,1,0,1,2,4,3,3,2,1',
                '4,3,3,2,1,0,1,4,3,3,2,2,1,0,1,4,3,3,2,2','4,3,3,2,2,1,1,0,0,4,3,3,2,2,1,1,0,1,2,3','4,3,3,2,2,1,1,0,1,0,4,3,3,2,2,1,1,0,1,2','4,3,3,2,1,0,1,4,3,3,2,2,1,0,1,4,3,3,2,2,1','4,3,3,2,2,1,0,4,3,3,2,2,1,0,4,3,3,2,2,1,1',
                '4,3,3,2,2,1,1,0,1,0,4,3,3,2,2,1,1,0,1,2,3','4,3,2,1,0,1,2,4,3,3,2,1,0,1,2,4,3,3,2,1,0','4,3,3,2,2,1,1,0,1,2,4,3,3,2,2,1,1,0,1,2,0','4,3,3,2,2,1,0,4,3,3,2,2,1,1,0,4,3,3,2,2,1,1','4,3,3,2,2,1,1,0,1,2,0,4,3,3,2,2,1,1,0,1,2,3',
                '4,3,3,2,1,0,1,4,3,3,2,2,1,0,1,4,3,3,2,2,1,0','4,3,2,1,0,1,2,4,3,3,2,1,0,1,2,4,3,3,2,1,0,1','4,3,3,2,2,1,1,4,3,3,2,2,1,1,0,4,3,3,2,2,1,1,0','4,3,3,2,2,1,0,4,3,3,2,2,1,1,0,4,3,3,2,2,1,1,0','4,3,3,2,1,0,1,4,3,3,2,2,1,0,1,4,3,3,2,2,1,0,1',
                '4,3,2,1,0,1,2,4,3,3,2,1,0,1,2,4,3,3,2,1,0,1,2','4,3,3,2,2,1,1,0,1,2,3,4,3,3,2,2,1,1,0,1,2,3,0','4,3,3,2,1,0,1,4,3,3,2,2,1,0,1,4,3,3,2,2,1,0,1,2',
                '4,3,2,1,0,1,2,4,3,3,2,1,0,1,2,4,3,3,2,1,0,1,2,3','4,3,3,2,2,1,0,4,3,3,2,2,1,1,0,4,3,3,2,2,1,1,0,1','4,3,3,2,2,1,1,4,3,3,2,2,1,1,0,4,3,3,2,2,1,1,0,0'
            ]
        )
        c = rootSystem(arr,'C')
        c.generate_up_to_height(3*c.deltaHeight)
        self.assertSetEqual(toSet(c),set(expected))
    def test_m_k(self):
        arr = [4,0,1,2,3]
        c = rootSystem(arr,'C')
        c.generate_up_to_delta(1)
        self.assertEqual(c.m_k([0,1,0,0,0]),2)
    def test_C_Affine_2(self):
        arr = [3,0,1,2]
        expected = set(
            ['3','0','1','2','3,2','1,2','0,1', '3,2,1','3,2,2','0,1,2','0,1,1','3,2,2,1','0,1,2,1','3,2,1,0','3,2,2,1,1',
            '3,2,2,1,0','3,2,1,0,1','0,1,2,1,2','3,2,2,1,1,0','3,2,2,1,0,1','3,2,1,0,1,2','3,2,2,1,1,0,1','3,2,2,1,0,1,2',
            '3,2,1,3,2,1,0','3,2,2,1,1,0,0','3,2,1,0,1,3,2,2','3,2,2,1,1,0,1,2','3,2,2,1,1,0,1,0','3,2,1,0,1,3,2,2,1',
            '3,2,2,1,3,2,2,1,0','3,2,2,1,1,0,1,2,0','3,2,2,1,1,0,1,0,1','3,2,2,1,0,3,2,2,1,1','3,2,2,1,1,0,1,2,0,1',
            '3,2,1,0,1,3,2,2,1,0','3,2,2,1,1,3,2,2,1,1,0','3,2,2,1,0,3,2,2,1,1,0','3,2,1,0,1,3,2,2,1,0,1','3,2,2,1,1,0,1,2,0,1,2',
            '3,2,2,1,1,3,2,2,1,1,0,0','3,2,2,1,0,3,2,2,1,1,0,1','3,2,1,0,1,3,2,2,1,0,1,2','3,2,2,1,1,0,3,2,2,1,1,0,1','3,2,2,1,0,1,3,2,2,1,0,1,2',
            '3,2,1,0,1,3,2,1,0,1,3,2,2','3,2,2,1,1,0,3,2,2,1,1,0,0','3,2,1,0,1,3,2,2,1,0,1,3,2,2','3,2,2,1,1,0,3,2,2,1,1,0,1,2','3,2,2,1,1,0,0,3,2,2,1,1,0,1',
            '3,2,1,0,1,3,2,2,1,0,1,3,2,2,1','3,2,2,1,0,3,2,2,1,0,3,2,2,1,1','3,2,2,1,1,0,0,3,2,2,1,1,0,1,2','3,2,2,1,1,0,1,3,2,2,1,1,0,1,0','3,2,2,1,0,3,2,2,1,1,0,3,2,2,1,1',
            '3,2,2,1,1,0,1,0,3,2,2,1,1,0,1,2','3,2,1,0,1,3,2,2,1,0,1,3,2,2,1,0','3,2,2,1,1,3,2,2,1,1,0,3,2,2,1,1,0','3,2,2,1,0,3,2,2,1,1,0,3,2,2,1,1,0',
            '3,2,1,0,1,3,2,2,1,0,1,3,2,2,1,0,1','3,2,2,1,1,0,1,2,3,2,2,1,1,0,1,2,0','3,2,1,0,1,3,2,2,1,0,1,3,2,2,1,0,1,2',
            '3,2,2,1,0,3,2,2,1,1,0,3,2,2,1,1,0,1','3,2,2,1,1,3,2,2,1,1,0,3,2,2,1,1,0,0'
]
        )
        c = rootSystem(arr,'C')
        c.generate_up_to_height(3*c.deltaHeight)
        self.assertSetEqual(toSet(c),set(expected))
    '''def test_A_Affine_1(self):
        arr=[1,0]
        expected = set(['1','0','1,0','1,1,0','1,0,0','1,1,0,0',
                        '1,0,1,0,0','1,1,0,1,0','1,1,0,1,0,0','1,1,0,1,0,1,0',
                        '1,0,1,0,1,0,0','1,1,0,1,0,1,0,0'])
        self.assertEqual(getSet("A",arr,affineCount=3),expected)'''
    def test_A_Affine_2(self):
        arr=[1,2,0]
        expected = set(
            ['1','2','0','1,2','1,0','2,0','1,0,2','1,2,0',
             '1,2,1,0','1,0,2,2','1,0,2,0','1,2,1,0,2','1,0,1,0,2',
             '1,0,2,0,2','1,0,1,0,2,2','1,2,1,0,2,0',
             '1,2,1,0,2,1,0','1,0,2,1,0,2,2','1,0,2,1,0,2,0',
             '1,2,1,0,2,1,0,2','1,0,1,0,2,1,0,2','1,0,2,2,1,0,2,0',
             '1,0,1,0,2,1,0,2,2','1,2,1,0,2,1,0,2,0']
        )
        a = rootSystem(arr,'A')
        a.generate_up_to_height(3*a.deltaHeight)
        self.assertEqual(toSet(a),expected)
    def test_A_Affine_3(self):
        arr=[1,2,3,4,0]
        expected = set(
            ['1','2','3','4','0','1,2','2,3','3,4','4,0',
             '1,0','1,2,3','2,3,4','3,4,0','1,0,4','1,0,2',
             '1,2,3,4','2,3,4,0','1,0,4,3','1,0,4,2','1,0,2,3',
             '1,0,4,3,2','1,0,2,3,4','1,2,3,4,0','1,2,3,4,1,0',
             '1,2,3,4,1,0,2','1,2,3,4,1,0,2,3','1,2,3,4,1,0,2,3,4',
             '1,0,4,3,2,2','1,0,4,3,2,3,2','1,0,4,3,2,3,4,2',
             '1,0,4,3,2,3,4,0,2','1,0,4,3,2,3','1,0,4,3,2,3,4',
             '1,0,4,2,3,4','1,0,4,3,2,3,4,0','1,0,4,2,3','1,0,2,3,4,0',
             '1,0,2,3,1,0,4,2','1,0,4,2,1,0,4,3,2','1,2,3,4,1,0,2,3,4,0',
             '1,0,2,3,1,0,4,2,3','1,0,4,2,1,0,4,3,2,3','1,0,2,3,1,0,4,2,3,4',
             '1,0,4,2,3,4,0','1,0,4,2,1,0,4,3','1,0,4,3,1,0,4,3,2',
             '1,0,4,3,1,0,4,3,2,2','1,0,2,3,1,0,4'
             ]
        )
        a = rootSystem(arr,'A')
        a.generate_up_to_height(2*a.deltaHeight)
        self.assertEqual(toSet(a),expected)
    def test_A_Affine_4(self):
        arr=[1,0,2,3]
        expected = set(['1','2','3','0','1,2','2,3','0,3','1,0','1,2,3,0'
                        ,'1,2,3','0,3,2','1,0,3','1,2,0','1,2,3','1,2,0,3','1,0,3,2'
                        ,'1,0,3,1,2','1,2,0,3,2','1,2,3,0,3','1,2,3,0,0','1,2,0,1,2,3',
                        '1,2,3,0,3,2','1,2,3,0,3,0','1,0,3,1,2,0','1,2,3,1,2,3,0','1,2,3,0,3,2,0','1,0,3,1,2,0,3',
                        '1,2,0,1,2,3,0','1,2,3,1,2,3,0,0','1,2,0,1,2,3,0,3','1,0,3,1,2,0,3,2',
                        '1,0,3,1,2,0,3,1,2','1,2,0,3,1,2,0,3,2','1,2,3,0,1,2,3,0,3','1,2,3,0,1,2,3,0,0',
                        '1,2,0,1,2,3,0,1,2,3','1,2,3,0,1,2,3,0,3,2','1,2,3,0,0,1,2,3,0,3',
                        '1,0,3,1,2,0,3,1,2,0','1,2,3,1,2,3,0,1,2,3,0','1,2,3,0,0,1,2,3,0,3,2',
                        '1,0,3,1,2,0,3,1,2,0,3','1,2,0,1,2,3,0,1,2,3,0','1,2,3,1,2,3,0,1,2,3,0,0',
                        '1,2,0,1,2,3,0,1,2,3,0,3','1,0,3,1,2,0,3,1,2,0,3,2'])
        a = rootSystem(arr,'A')
        a.generate_up_to_height(a.deltaHeight * 3)
        self.assertEqual(toSet(a),expected)
    def test_A_Affine_5(self):
        arr=[1,2,3,4,5,0]
        expected = set(['1','2','3','4','5','0','1,2','4,5','2,3','3,4','5,0','1,0','1,2,3','2,3,4','3,4,5','4,5,0',
                        '1,0,5','1,0,2','1,2,3,4','2,3,4,5','3,4,5,0','1,0,5,4','1,0,5,2','1,0,2,3','1,0,5,2,3',
                        '1,2,3,4,5','2,3,4,5,0','1,0,5,4,3','1,0,5,4,2','1,0,2,3,4','1,0,5,4,3,2','1,0,5,4,2,3',
                        '1,0,5,2,3,4','1,0,2,3,4,5','1,2,3,4,5,0','1,2,3,4,5,1,0','1,0,5,4,3,2,2','1,0,5,4,3,2,3',
                        '1,0,5,4,2,3,4','1,0,5,2,3,4,5','1,0,2,3,4,5,0','1,2,3,4,5,1,0,2','1,0,5,4,3,2,3,2',
                        '1,0,5,4,3,2,3,4','1,0,5,4,2,3,4,5','1,0,5,2,3,4,5,0','1,0,2,3,4,1,0,5','1,2,3,4,5,1,0,2,3',
                        '1,0,5,4,3,2,3,4,2','1,0,5,4,3,2,3,4,5','1,0,5,4,2,3,4,5,0','1,0,5,2,3,1,0,5,4','1,0,2,3,4,1,0,5,2',
                        '1,2,3,4,5,1,0,2,3,4','1,0,5,4,3,2,3,4,5,2','1,0,5,4,3,2,3,4,5,0','1,0,5,4,2,1,0,5,4,3',
                        '1,0,5,4,2,1,0,5,4,3','1,0,5,2,3,1,0,5,4,2','1,0,2,3,4,1,0,5,2,3','1,2,3,4,5,1,0,2,3,4,5',
                        '1,0,5,4,3,2,3,4,5,0,2','1,0,5,4,3,1,0,5,4,3,2','1,0,5,4,2,1,0,5,4,3,2','1,0,5,2,3,1,0,5,4,2,3',
                        '1,0,2,3,4,1,0,5,2,3,4','1,0,5,4,3,1,0,5,4,3,2,2','1,0,5,4,2,1,0,5,4,3,2,3','1,0,5,2,3,1,0,5,4,2,3,4',
                        '1,0,2,3,4,1,0,5,2,3,4,5','1,2,3,4,5,1,0,2,3,4,5,0'])
        a = rootSystem(arr,'A')
        a.generate_up_to_height(2*a.deltaHeight)
        self.assertEqual(toSet(a),expected)
    def test_D_Affine_1(self):
        arr=[0,1,2,3,4]
        expected=set(['0','1','2','3','4','1,2','2,3','2,4','0,2','1,2,3','1,2,4','2,4,3','0,2,4','0,2,3','0,2,1','1,2,4,3',
                      '0,2,4,3','0,2,4,1','0,2,3,1','1,2,4,3,2','0,2,4,3,2','0,2,4,3,1','0,2,4,1,2','0,2,3,1,2','0,2,4,3,2,1',
                      '0,2,4,3,1,2','0,2,4,1,2,3','0,2,3,1,2,4','0,2,4,3,2,1,1','0,2,4,3,2,1,2','0,2,4,3,1,2,3','0,2,4,3,1,2,4',
                      '0,2,3,1,0,2,4','0,2,4,3,2,1,2,1','0,2,4,3,2,1,2,3','0,2,4,3,2,1,2,4','0,2,3,1,2,0,2,4','0,2,4,3,2,1,2,3,1',
                      '0,2,4,3,2,1,2,4,1','0,2,4,3,2,1,2,4,3','0,2,4,1,2,0,2,4,3','0,2,3,1,2,0,2,4,3','0,2,3,1,2,0,2,4,1','0,2,4,3,2,1,2,4,3,1',
                      '0,2,4,3,1,0,2,4,3,2','0,2,4,1,2,0,2,4,3,1','0,2,3,1,2,0,2,4,3,1','0,2,4,3,2,1,2,4,3,1,2','0,2,4,3,2,0,2,4,3,2,1',
                      '0,2,4,3,1,0,2,4,3,2,1','0,2,4,1,2,0,2,4,3,1,2','0,2,3,1,2,0,2,4,3,1,2','0,2,4,3,2,0,2,4,3,2,1,1','0,2,4,3,1,0,2,4,3,2,1,2',
                      '0,2,4,1,2,0,2,4,3,1,2,3','0,2,3,1,2,0,2,4,3,1,2,4','0,2,4,3,2,1,0,2,4,3,2,1,1','0,2,4,3,2,1,0,2,4,3,2,1,2','0,2,4,3,1,2,0,2,4,3,1,2,3',
                      '0,2,4,3,1,2,0,2,4,3,1,2,4','0,2,3,1,2,0,2,4,3,1,0,2,4','0,2,4,3,2,1,1,0,2,4,3,2,1,2','0,2,4,3,2,1,0,2,4,3,2,1,2,3',
                      '0,2,4,3,2,1,0,2,4,3,2,1,2,4','0,2,3,1,2,0,2,4,3,1,2,0,2,4','0,2,4,3,2,1,1,0,2,4,3,2,1,2,3','0,2,4,3,2,1,1,0,2,4,3,2,1,2,4',
                      '0,2,4,3,2,1,0,2,4,3,2,1,2,4,3','0,2,4,1,2,0,2,4,3,1,2,0,2,4,3','0,2,3,1,2,0,2,4,3,1,2,0,2,4,3','0,2,3,1,2,0,2,4,3,1,2,0,2,4,1',
                      '0,2,4,3,2,1,1,0,2,4,3,2,1,2,4,3','0,2,4,3,1,0,2,4,3,2,1,0,2,4,3,2','0,2,4,1,2,0,2,4,3,1,2,0,2,4,3,1','0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1',
                      '0,2,4,3,2,1,2,3,1,0,2,4,3,2,1,2,4','0,2,4,3,2,0,2,4,3,2,1,0,2,4,3,2,1','0,2,4,3,1,0,2,4,3,2,1,0,2,4,3,2,1','0,2,4,1,2,0,2,4,3,1,2,0,2,4,3,1,2',
                      '0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2','0,2,4,3,2,0,2,4,3,2,1,0,2,4,3,2,1,1','0,2,4,3,1,0,2,4,3,2,1,0,2,4,3,2,1,2','0,2,4,1,2,0,2,4,3,1,2,0,2,4,3,1,2,3',
                      '0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,4','0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2,1,1','0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2,1,2','0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,3',
                      '0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,4','0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1,0,2,4','0,2,4,3,2,1,0,2,4,3,2,1,2,0,2,4,3,2,1,1',
                      '0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2,1,2,3','0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2,1,2,4','0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4',
                      '0,2,4,3,2,1,0,2,4,3,2,1,2,3,0,2,4,3,2,1,1','0,2,4,3,2,1,0,2,4,3,2,1,2,4,0,2,4,3,2,1,1','0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2,1,2,4,3',
                      '0,2,4,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3','0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3','0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,1',
                      '0,2,4,3,2,1,0,2,4,3,2,1,2,4,3,0,2,4,3,2,1,1','0,2,4,3,1,0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2','0,2,4,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3,1'
                      ,'0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3,1','0,2,4,3,2,1,1,0,2,4,3,2,1,2,4,3,0,2,4,3,2,1,2','0,2,4,3,2,0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2,1',
                      '0,2,4,3,1,0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2,1','0,2,4,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2','0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2',
                      '0,2,4,3,2,0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2,1,1','0,2,4,3,1,0,2,4,3,2,1,0,2,4,3,2,1,0,2,4,3,2,1,2','0,2,4,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,3',
                      '0,2,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,0,2,4,3,1,2,4'])
        D = rootSystem(arr,'D')
        D.generate_up_to_height(4*D.deltaHeight)
        self.assertEqual(toSet(D),expected)
    def test_D_Affine_2(self):
        arr = [1,4,5,2,3,0]
        expected = set(['1','2','3','4','5','0','1,2','2,3','4,3','5,3','2,0','1,2,3','4,3,2','5,3,2',
            '4,3,5','2,0,3','1,2,0','1,2,3,4','1,2,3,5','4,3,2,5','5,3,2,0','4,3,2,0','1,2,0,3','1,2,3,5,4','4,3,2,5,3',
            '4,3,2,0,5','1,2,0,3,5','1,2,0,3,4','1,2,0,3,2','1,2,3,5,4,3','4,3,2,0,5,3','1,2,0,3,5,4','1,2,0,3,2,5','1,2,0,3,2,4',
            '1,2,3,5,4,3,2','4,3,2,0,5,3,2','1,2,0,3,5,4,3','1,2,0,3,2,5,4','1,2,0,3,2,5,3','1,2,0,3,2,4,3','1,2,0,3,2,5,3,4','1,2,0,3,2,5,4,3',
            '1,2,0,3,2,4,3,5','1,2,0,3,5,4,3,2','1,2,3,5,4,3,2,0','1,2,3,5,4,3,1,2,0','1,2,0,3,2,5,4,3,2','1,2,0,3,2,5,3,4,3','1,2,0,3,2,5,3,4,4','1,2,0,3,2,5,4,3,5',
            '1,2,0,3,5,4,3,2,0','1,2,3,5,4,3,2,1,2,0','1,2,0,3,2,5,3,4,3,2','1,2,0,3,2,5,3,4,3,4','1,2,0,3,2,5,3,4,3,5','1,2,0,3,2,5,4,3,2,0','1,2,3,5,4,3,2,1,2,0,3',
            '1,2,0,3,2,5,3,4,3,2,4','1,2,0,3,2,5,3,4,3,2,5','1,2,0,3,2,5,3,4,3,5,4','1,2,0,3,2,5,3,4,3,2,0','1,2,0,3,5,4,1,2,0,3,2','1,2,3,5,4,3,2,1,2,0,3,4',
            '1,2,3,5,4,3,2,1,2,0,3,5','1,2,0,3,2,5,3,4,3,2,5,4','1,2,0,3,2,5,3,4,3,2,0,5','1,2,0,3,2,5,3,4,3,2,0,4','1,2,0,3,5,4,3,1,2,0,3,2','1,2,3,5,4,3,2,1,2,0,3,5,4',
            '1,2,0,3,2,5,3,4,3,2,5,4,3','1,2,0,3,2,5,3,4,3,2,0,5,4','1,2,0,3,5,4,3,1,2,0,3,2,5','1,2,0,3,5,4,3,1,2,0,3,2,4','1,2,0,3,2,4,3,1,2,0,3,2,5','1,2,3,5,4,3,2,1,2,0,3,5,4,3',
            '1,2,0,3,2,5,3,4,3,2,0,5,4,3','1,2,0,3,5,4,3,1,2,0,3,2,5,4','1,2,0,3,2,5,4,1,2,0,3,2,5,3','1,2,0,3,2,4,3,1,2,0,3,2,5,4','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2',
            '1,2,0,3,2,5,3,4,3,2,0,5,4,3,2','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3','1,2,0,3,2,5,4,1,2,0,3,2,5,3,4','1,2,0,3,2,5,3,1,2,0,3,2,5,3,4','1,2,0,3,2,4,3,1,2,0,3,2,5,4,3',
            '1,2,0,3,2,5,3,1,2,0,3,2,5,3,4,4','1,2,0,3,2,5,4,1,2,0,3,2,5,3,4,3','1,2,0,3,2,4,3,1,2,0,3,2,5,4,3,5','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,2','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,0',
            '1,2,3,5,4,3,2,1,2,0,3,5,4,3,1,2,0','1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,2','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,4','1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,5',
            '1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,0','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2','1,2,0,3,2,5,3,4,4,1,2,0,3,2,5,3,4,3','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,5',
            '1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,2,0','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3','1,2,0,3,2,5,3,4,4,1,2,0,3,2,5,3,4,3,2','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,5','1,2,0,3,2,5,3,4,4,1,2,0,3,2,5,3,4,3,5',
            '1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,0','1,2,0,3,5,4,3,1,2,0,3,2,5,4,1,2,0,3,2','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,4','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5','1,2,0,3,2,5,3,4,4,1,2,0,3,2,5,3,4,3,2,5',
            '1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,0,5','1,2,0,3,2,5,3,4,4,1,2,0,3,2,5,3,4,3,2,0','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4',
            '1,2,0,3,2,5,3,4,3,5,4,1,2,0,3,2,5,3,4,3,2','1,2,0,3,2,5,3,4,4,1,2,0,3,2,5,3,4,3,2,0,5','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,4',
            '1,2,0,3,2,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3','1,2,0,3,2,5,3,4,3,5,4,1,2,0,3,2,5,3,4,3,2,0','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4',
            '1,2,0,3,2,5,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3','1,2,0,3,2,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2','1,2,0,3,2,5,3,4,3,2,5,4,1,2,0,3,2,5,3,4,3,2,0',
            '1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3','1,2,0,3,2,5,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4','1,2,0,3,2,5,3,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4','1,2,0,3,2,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3',
            '1,2,0,3,2,5,3,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,4','1,2,0,3,2,5,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3','1,2,0,3,2,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,5','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,2',
            '1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,0','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,1,2,0','1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,2','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3',
            '1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,4','1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,5','1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,0','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0',
            '1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,1,2,0,3,2,5,3,4,4','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,5','1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,2,0',
            '1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,1,2,0,3,2,5,3,4,4','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,5','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,5,1,2,0,3,2,5,3,4,4',
            '1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,0','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,1,2,0,3,2','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,4','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5',
            '1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,5,1,2,0,3,2,5,3,4,4','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,0,5','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,0,1,2,0,3,2,5,3,4,4','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2',
            '1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4','1,2,0,3,2,5,3,4,4,1,2,0,3,2,5,3,4,3,2,5,1,2,0,3,2,5,3,4,3','1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3,2,0,5,1,2,0,3,2,5,3,4,4','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5',
            '1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,4','1,2,0,3,2,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3','1,2,0,3,2,5,3,4,4,1,2,0,3,2,5,3,4,3,2,0,5,1,2,0,3,2,5,3,4,3',
            '1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4','1,2,0,3,2,5,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3','1,2,0,3,2,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2',
            '1,2,0,3,2,5,3,4,4,1,2,0,3,2,5,3,4,3,2,0,5,1,2,0,3,2,5,3,4,3,2','1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3','1,2,0,3,2,5,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4','1,2,0,3,2,5,3,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4',
            '1,2,0,3,2,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3','1,2,0,3,2,5,3,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,4','1,2,0,3,2,5,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,1,2,0,3,2,5,3,4,3','1,2,0,3,2,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,5',
            '1,2,0,3,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,1,2,0,3,2,5,4,3,2','1,2,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,1,2,0,3,5,4,3,2,0',])
        a = rootSystem(arr,'D')
        a.generate_up_to_height(4*a.deltaHeight)
        self.assertEqual(toSet(a),expected)
    def test_D_Affine_3(self):
        D4 = rootSystem([0,4,1,2,3],'D')
        D4.SL([2,2,5,2,3])
    def test_F_Affine_1(self):
        arr = [1,2,3,4,0]
        r = rootSystem(arr,'F')
if __name__ == '__main__':
    unittest.main()