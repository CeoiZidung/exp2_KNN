import numpy as np

class Node(object):
    def __init__(self,left=None,right=None,father=None,item=None,label=None,depth=None,divided_dimension=None):
        """
        树中的结点对象
        参数：
        left：左孩子
        right：右孩子
        father：父节点
        item：该节点的特征坐标，如有n个特征，则为1*n的向量
        lable：该点所属类别
        depth：位于KD的第几层，0，1，2...
        divided_dimension：依据第几维划分超平面
        """
        self.left=left
        self.right=right
        self.item=item
        self.label=label
        self.depth=depth
        self.divided_dimension=divided_dimension
        self.father=father

class KDtree(object):
    """
    KD树的构建
    """
    def __init__(self,X,Y,dis_mes='2',fea_sec='index'):
        """
        :param X: m*n矩阵，m为点个数，n为特征个数
        :param Y: m*1向量，标签
        :param dis_mea: 距离度量范数 1 2 inf
        :param fea_sec: 挑选超平面 index :depth%n  var :选取最大方差那一列
        """
        self.__length=0
        self.dis_mes=dis_mes
        self.fea_sec=fea_sec
        #开始建树
        self.root=self.__create(X,Y,0)



    def __find_delete_point(self,target):
        '''
        先从树中寻找有无target，没有返回False，找到了返回Node_target
        :param target:  待删除的点
        :return:
        '''
        target=np.array(target)
        node=self.root
        while True:
            if node is None:    #如果为None，说明查找失败
                return False
            #本来两者是np数据，比较会报错，将其变为list比较
            if list(node.item)==list(target):   #找到该点
                return node
            else:   #该节点不是所要找的删除点，往该节点的左or右子树找
                if node.divided_dimension is None:
                    #当node没有划分平面，说明该点是叶子节点，说明查找失败
                    return False

                div_dim=node.divided_dimension  #该点有划分平面，不是叶子节点
                if target[div_dim]<=node.item[div_dim]:  #进入左节点递归
                    node=node.left
                else:
                    node=node.right

    def find_del(self,node):
        #如果待删除节点为叶子节点，直接删除
        if node.left is None and node.right is None:
                #这个if循环是为了确定node节点是father节点的左还是右节点
                if node.father.left==node:
                    node.father.left=None
                else:
                    node.father.right=None
                del node    #销毁node节点
        elif node.right is None:    #仅有左子树
            if node.father.left==node:
                node.father.left=node.left
            else:
                node.father.right=node.left
            #把待删除点node的father链接到node的左节点上
            node.left.father=node.father
            del node
        elif node.left is None: #仅有右子树
            if node.father.left==node:
                node.father.left=node.right
            else:
                node.father.right=node.right
            #把待删除点node的father链接到node的右节点上
            node.right.father=node.father
            del node

        else:   #左右节点都有树
            #遍历左子树，找到左节点中cut_dim列特征中最大的node
            cut_dim=node.divided_dimension
            lis_node=self.show(node.left)   #层序遍历左节点
            lis_cut_dim=[nd.item[cut_dim] for nd in lis_node]   #左子树所有节点的cut_dim维特征比较
            max_index=lis_cut_dim.index(max(lis_cut_dim))   #定位最大那个节点所在位置
            tnode=lis_node[max_index]   #用它替代待删除的节点
            #这两步变相于把node删除了。
            node.item=tnode.item
            node.label=tnode.label
            self.find_del(tnode)    #tnode替代了他的位置，就相当它从原位置被删除了，所以继续进行find_del

    def delete(self, target):
        node=self.__find_delete_point(target)   #要删除先得找到该点
        if node==False:     #该点不存在
            print('target point doesnot exist!')
            return
        else:
            self.find_del(node)

    def insert(self,target):
        n=target.item.shape[0]
        node=self.__DFT(target.item,self.root)[-1]
        target.depth=node.depth+1
        target.father=node
        if node.divided_dimension is None:
            if self.fea_sec=='index':
                #切分平面为depth_i%n
                max_index=node.depth%n
            else:
                X=np.array([node.item,target.item])
                #选取方差最大的特征作为切分平面
                vars=[np.var(X[:,col]) for col in range(n)]
                max_index=vars.index(max(vars)) #方差最大特征对应列数
            node.divided_dimension=max_index
            if node.item[max_index]>target.item[max_index]:
                node.left=target
            else:
                node.right=target
        else:
            if node.left is None:
                node.left=target
            else:   node.right=target

    def __cal_dist(self,vector1,vector2):
        if self.dis_mes=='2':   #欧氏距离
            op=np.sqrt(np.sum(np.square(vector1-vector2)))
        elif self.dis_mes=='1': #曼哈顿距离
            op=np.sum(np.abs(vector1-vector2))
        else:   #self.dis_mes=inf
            op=np.max(np.abs(vector1-vector2))

        return op

    def show(self,root): #层序遍历展示，return一个列表
        output=[]
        if not root :
            return output
        myQueue = []
        node = root
        myQueue.append(node)
        while myQueue:
            node = myQueue.pop(0)
            output.append(node)
            if node.left:
                myQueue.append(node.left)
            if node.right:
                myQueue.append(node.right)

        return output

    def __create(self,X,Y,depth_i):
        '''

        :param X:  待建树的特征为数据m*n矩阵，m为数据个数，n为特征维度
        :param Y:   m*1向量，为数据对应的label
        :param depth_i:    位于第几层
        :return:    node.left、right递归建树后，返回node
        '''
        X=np.array(X)
        Y=np.array(Y)
        m,n=X.shape
        if m==0:
            return None
        elif m==1:
            self.__length+=1
            node=Node(item=X[0],label=Y[0],depth=depth_i)     #X[0]表示第一行，仍是向量
            return node
        else:   #m>2 至少一个左节点
            if self.fea_sec=='index':
                #切分平面为depth_i%n
                max_index=depth_i%n
                max_feather_sorted_index=X[:,max_index].argsort()
                divided_node_index=max_feather_sorted_index[m//2]
            else:   #self.fea_sec=='var'
                #选取方差最大的特征作为切分平面
                vars=[np.var(X[:,col]) for col in range(n)]
                max_index=vars.index(max(vars)) #方差最大特征对应列数
                max_feather_sorted_index=X[:,max_index].argsort()
                divided_node_index=max_feather_sorted_index[m//2]

            node=Node(item=X[divided_node_index],label=Y[divided_node_index],divided_dimension=max_index,depth=depth_i)
            self.__length+=1
            #左孩子
            left_X_index=max_feather_sorted_index[0:m//2]
            node.left=self.__create(X[left_X_index],Y[left_X_index],depth_i+1)
            node.left.father=node
            #右孩子
            if not m==2:   #即m==2时没有右孩子
                right_X_index=max_feather_sorted_index[m//2+1:]
                node.right=self.__create(X[right_X_index],Y[right_X_index],depth_i+1)
                node.right.father=node

            return node

    def __DFT(self,target,root_subtree):
        """
        :param target: 目标点
        :param root_subtree: 子树
        :return: 遍历过得节点列表
        """
        stack=[]
        if root_subtree is None:
            return stack    #空列表
        node=root_subtree
        while True: #找到距离target最近的节点tnode
            stack.append(node)
            if node.divided_dimension is None:  #节点没有这个属性，为None
                break

            div_dim=node.divided_dimension
            if target[div_dim]<=node.item[div_dim]:  #进入左节点递归
                if node.left is not None:
                    node=node.left
                else:   break
            else:
                if node.right != None:
                    node=node.right
                else:   break

        return stack

    def find_nearest(self,target):
        """
        寻找point在KD树中的最近邻点，先找叶节点，在回溯
        :param target: n*1向量，该点的坐标（特征）
        :return: [最近邻点、两点距离]
        """
        stack=[]    #回溯用到
        target=np.array(target)
        ######################找叶节点#######################################
        if self.root==None:
            return [None,None]
        stack=self.__DFT(target,self.root)

        ######################回溯找到最近点，直到stack为空#####################
        d=None
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
                #判断半径d与超平面是否相交
                #即判断target到超平面距离d_v与d的大小
                d_v=abs(target[tp.divided_dimension]-tp.item[tp.divided_dimension])
                if d_v<d:
                   if target[tp.divided_dimension]<=tp.item[tp.divided_dimension]:
                       #说明该target位于左边，所以another_child是右孩子
                       another_child=tp.right
                   else: another_child=tp.left
                   stack=stack+self.__DFT(target,another_child)


        return [d,nearest_point]










