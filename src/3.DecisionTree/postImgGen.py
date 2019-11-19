#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib
import random
import time

# 背景颜色list
colorList = ['#e96060', '#e95555', '#d63939', '#fd5c5c', '#d55f5f']
footerStrList = {0: '江西专升本社区:jxzsb.club', 1: '来自 《春雏集》', 3: ''}

# 输入文章标题
def genPostImg(title,index):
    """
        dec:
            生成文章配图

            春雏集类型的内容，有的时候会有很长的内容，
            上下留空为300px ， 总长度为 = 300 * 2 + 字行数 * 75 + (字行数 - 1)*150
            fontsize = 60 时，字长 = 75 px ，字宽 = 80px
        pram：
            title -- 文字内容
            index -- 类型
        usage：

    """
    # 中文显示乱码 设置字体
    myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttc')  # 这一行
    #  行的初始坐标，默认是左下角
    x, y = 0, 7


    # 如果title 字数过多，或者需要分行显示，且居中  最多显示两行，title 字符不超过27个字
    title_lines = title.split(' ')
    liens = len(title_lines)
    print(title_lines)
    chunchu_lens = int(liens * 75 +600 + (liens -1)*75 )/100
    print("chunchu_lens=",chunchu_lens)
    #
    if chunchu_lens <= 10 :
        fig = plt.figure(figsize=(23, 10), facecolor=colorList[random.randint(0, len(colorList) - 1)], edgecolor='#ffffff')
        plt.axis([0, 23, 0, 10])
    else:
        fig = plt.figure(figsize=(23, chunchu_lens), facecolor=colorList[random.randint(0, len(colorList) - 1)],
                         edgecolor='#ffffff')
        plt.axis([0, 23, 0, chunchu_lens])
    # 去除留白
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)



    #plt.text(2.5, 5, title, color='#ffffff', style='italic', va='center', wrap=True, fontproperties=myfont,fontsize=60, )
    #plt.text(1.5, 2, footerStrList.get(index), color='#ffffff', style='italic', wrap=True, fontproperties=myfont,
    #         fontsize=40, )
    if index == 0:
        for i in range(len(title_lines)):
            x = (23 - (len(title_lines[i]) * 80 / 100)) / 2
            print(x)
            y -= 1.5
            plt.text(x, y, title_lines[i], color='#ffffff', style='italic', wrap=True, fontproperties=myfont,
                     fontsize=60, )
        plt.text(1.5, 2, footerStrList.get(index), color='#ffffff', style='italic', wrap=True, fontproperties=myfont,
                 fontsize=40, )
    if index == 1:
        y = chunchu_lens - 2
        for i in range(len(title_lines)):
            x = (23 - (len(title_lines[i]) * 80 / 100)) / 2
            print(x)
            y -= 1.5
            plt.text(x, y, title_lines[i], color='#ffffff', style='italic', wrap=True, fontproperties=myfont,
                     fontsize=60, )
        plt.text(15, 2, footerStrList.get(index), color='#ffffff', style='italic', wrap=True, fontproperties=myfont,
                 fontsize=40, )
        plt.text(16, 1.2, time.strftime("%Y-%m-%d", time.localtime()), color='#ffffff', style='italic', wrap=True, fontproperties=myfont,
                 fontsize=40, )
    if index == 2:
        plt.text(1.5, 2, footerStrList.get(index), color='#ffffff', style='italic', wrap=True, fontproperties=myfont,
                 fontsize=40, )

   # 隐藏 x,y 的刻度
    plt.xticks([])
    plt.yticks([])
    # 隐藏 x ，y 轴
    plt.axis('off')
    # 截取 title 的前7个字符
    plt.savefig('./'+title[:7]+'.png', facecolor=colorList[random.randint(0, len(colorList) - 1)], color='#ffffff', bbox_inches='tight' )
    plt.show()


def testPostImg():
    title = input("请输入标题：")
    type = int(input("请输入类型：0：post  1：春雏集"))
    genPostImg(title, type)
def test():
    myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttc')  # 这一行

    fig = plt.figure(figsize=(23, 10), facecolor=colorList[random.randint(0, len(colorList)-1)],edgecolor='#ffffff')
    plt.axis([0, 10, 0, 10])
    t = ("This is a really long string that I'd rather have wrapped so that it "
         "doesn't go outside of the figure, but if it's long enough it will go "
         "off the top or bottom!")

   # plt.text(4.NativeBayes, 1, t, ha='left', wrap=True)
   # plt.text(6, 5, t, ha='left', wrap=True)
    #plt.text(5, 5, t, ha='right',  wrap=True)
    #plt.text(5, 10, t, fontsize=18, ha='center', va='top', wrap=True)
    plt.text(2.5, 5, "江西专升本社区-付出及收获", color='#ffffff',  va='center', wrap=True,fontproperties=myfont, fontsize=60,)
    plt.text(1.5, 2, "江西专升本社区:jxzsb.club", color='#ffffff',   wrap=True,fontproperties=myfont, fontsize=40,)

    #plt.text(-1, 0, t, ha='left', rotation=-15, wrap=True)
    # frame = plt.gca()
    # # y 轴不可见
    # frame.axes.get_yaxis().set_visible(False)
    # # x 轴不可见
    # frame.axes.get_xaxis().set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig('./test2.png', facecolor=colorList[random.randint(0, len(colorList)-1)], color='#ffffff')
    plt.show()


if __name__ == '__main__':
    testPostImg()

