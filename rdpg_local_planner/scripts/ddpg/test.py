#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import visdom

def line_test2():
    
    viz = visdom.Visdom(env="line test1")
 
    # Example for Latex Support
    viz.line(
        X=[1, 2, 3, 4],  # x坐标
        Y=[1, 4, 9, 16],  # y值
        win="line1",  # 窗口id
        name="test line1",  # 线条名称
        update=None,  # 已添加方式加入
        opts={
            'showlegend': True,  # 显示网格
            'title': "Demo line in Visdom",
            'xlabel': "x1",  # x轴标签
            'ylabel': "y1",  # y轴标签
        },
    )
 
    viz.line(
        X=[1, 2, 3, 4],
        Y=[0.5, 2, 4.5, 8],
        win="line1",
        name="test line2",
        update='append',
    )

line_test2()
