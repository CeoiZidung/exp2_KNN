   def find_nearest(self,target):
        ######################第一步 DFT寻找叶节点#######################################
        略
        ######################第二步 回溯找到最近点，直到stack为空#########################
        d=None  #最近距离初始化为None
        nearest_point=None
        while stack!=[]:
            tp=stack.pop()
            if d is None:
                d=self.__cal_dist(tp.item,target)
                nearest_point=tp
            elif self.__cal_dist(tp.item,target)<d:
                d=self.__cal_dist(tp.item,target)
                nearest_point=tp
            if tp.divided_dimension is not None:
                #首先判断该点是否为叶子节点、是，该部分跳过（因为没有超平面）
                #不是，判断半径d与超平面是否相交
                #即判断target到超平面距离d_v与d的大小
                d_v=abs(target[tp.divided_dimension]-tp.item[tp.divided_dimension]) #到超平面的距离
                if d_v<d:
                   if target[tp.divided_dimension]<=tp.item[tp.divided_dimension]:
                       #说明该target位于左边，所以another_child是右孩子
                       another_child=tp.right
                   else: another_child=tp.left
                   stack=stack+self.__DFT(target,another_child)

        return [d,nearest_point]
