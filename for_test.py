   def find_del(self,node):
        #如果待删除节点为叶子节点，直接删除
        if node.left is None and node.right is None:
                #这个if循环是为了确定node节点是father节点的左还是右节点
                if node.father.left==node:
                    node.father.left=None
                else:
                    node.father.right=None
                del node    #销毁node节点
        elif node.right is None:
            if node.father.left==node:
                node.father.left=node.left
            else:
                node.father.right=node.left
            #把待删除点node的father链接到node的左节点上
            node.left.father=node.father
            del node
        elif node.left is None:
           #同上面操作
        else:   #左右节点都有树
            #遍历左子树，找到左节点中cut_dim列特征中最大的node
            cut_dim=node.divided_dimension
            lis_node=self.show(node.left)   #层序遍历左节点
            lis_cut_dim=[nd.item[cut_dim] for nd in lis_node]   #左子树所有节点的cut_dim维特征比较
            max_index=lis_cut_dim.index(max(lis_cut_dim))   #定位最大那个节点所在位置
            tnode=lis_node[max_index]   #用它替代待删除的节点
            node.item=tnode.item
            node.label=tnode.label
            del node
            self.find_del(tnode)    #tnode替代了他的位置，就相当它从原位置被删除了，所以继续进行find_del
