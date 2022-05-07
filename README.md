Implements elimination tree, but right-looking potrf. 

For finding offsets in CSC format, a map is used, which I believe to be sub-optimal. If no fill in calculated in advance, could use unpacked and etree to update.