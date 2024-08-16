import os
import glob
import pandas as pd
import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.graphframe import GraphFrame
from hatchet.frame import Frame
from typing import List



class GPTLReader:

    def __init__(self, log_file_or_dir: str) -> None:
        self.log_file_or_dir = log_file_or_dir
        pass

    def read(self) -> GraphFrame:
        if os.path.isdir(self.log_file_or_dir):
            return self.read_log_dir(self.log_file_or_dir)
        else:
            return self.read_log_simple(self.log_file_or_dir)

    def read_log_simple(self, log_file: str) -> GraphFrame:
        with open(log_file, 'r') as gptl_file:
            lines = gptl_file.readlines()
            for index, line in enumerate(lines):
                if line.strip().startswith('timer_name'):
                    # print(line)
                    column_names = line.strip().split()
                    lines = lines[index + 1:]
                    break
            dicts = []
            stack = [None]
            root_graph_nodes = []
            for index, line in enumerate(lines):
                # split line by whitespace
                split_line = line.split()
                
                # If the line is empty, we are done
                if len(split_line) == 0:
                    lines = lines[:index]
                    break

                # Take care of spaces in the name
                while len(split_line) > len(column_names) + 1:
                    split_line[1] = split_line[1] + ' ' + split_line[2]
                    split_line.pop(2)
                
                # get name time and depth from the line
                timer_name = split_line[1]
                depth = int(split_line[0]) - 1
                time = float(split_line[2])
                
                # adjust the stack to the correct depth
                if depth < len(stack) - 1:
                    stack = stack[:depth + 1]
                
                # create frame and graph node
                frame_node = Frame({'name': timer_name})
                graph_node = Node(frame_node, parent=stack[-1])
                
                # add the graph node to the parent
                if(stack[-1] is not None):
                    parent: Node = stack[-1]
                    parent.add_child(graph_node)
                
                # create a dictionary for the node
                node_dict = {}
                node_dict['name'] = timer_name
                node_dict['node'] = graph_node
                node_dict['time'] = time

                # add the dictionary to the list of dictionaries
                dicts.append(node_dict)
                
                # add the current node to the stack
                stack.append(graph_node)
                
                # add the node to the root_graph_nodes if it is a root node
                if(depth == 0):
                    root_graph_nodes.append(graph_node)

            # Create the graph                
            graph = hatchet.graphframe.Graph(root_graph_nodes)
            graph.enumerate_traverse()

            # Create the dataframe
            df = pd.DataFrame(dicts)
            df = df.set_index('node')

            # Create the GraphFrame
            gf = hatchet.GraphFrame(graph, df, [], ['time'])
            return gf    
    
    def check_existing_node(self, name: str, nodes: List[Node]) -> Node:
        # print('checking for node', name, 'in', nodes)
        for root in nodes:
            if root.frame.get('name') == name:
                # print('found node', name)
                return root
        return None

    def read_single_log(self, log_file: str, start_rank: int, end_rank: int) -> pd.DataFrame:
        with open(log_file, 'r') as gptl_file:
            rank_range = (start_rank, end_rank)
            lines = gptl_file.readlines()
            # skip first part of the file
            for index, line in enumerate(lines):
                # we want to stop at line that looks like this
                # Stats for thread 0:
                if line.startswith('Stats for thread '):
                    lines = lines[index + 1:]
                    break
            # read column names
            # column_names = lines[0].strip().split()
            # # column_names = [name.strip() for name in column_names if name != '']
            # # manually adjust UTR Overhead column name
            # column_names[-2] = 'UTR Overhead'
            # column_names.pop(-1)
            # print(column_names)
            lines = lines[1:]
            dicts = []
            stack: List[Node] = []
            # read the data
            for line in lines:
                if line.strip() == '':
                    break
                node_dict = {}
                # first lets get the depth
                # check for the first " character
                for i in range(len(line)):
                    if line[i] == '"':
                        node_dict['depth'] = int(i/2) - 1
                        # print(node_dict['depth'])
                        break
                # print(line)
                # now get rid of the spaces in the beginning 
                line = line[((node_dict['depth'] + 1) * 2) + 1:]
                # print('line', line)
                # now we split the line by " to seperate the name from everything else
                split_line = line.split('"')
                # print('split line init', split_line)
                node_dict['name'] = split_line[0]
                # now we split the rest of the line by whitespace
                split_line = split_line[1].split()
                # print('col names', column_names)
                # print('split line', split_line)
                # now we go through the columns and add them to the dictionary
                # for i in range(len(column_names)):
                #     if column_names[i] in self.numeric_metric_names:
                #         # print('column name', column_names[i])
                #         # print('value', split_line[i])
                #         node_dict[column_names[i]] = float(split_line[i])
                #     else:
                #         node_dict[column_names[i]] = split_line[i]
                # print('node dict', node_dict)

                # adjust the stack to the correct depth
                if node_dict['depth'] < len(stack):
                    stack = stack[:node_dict['depth']]

                # check if this node already exists
                graph_node = None
                if node_dict['depth'] == 0:
                    # print('looking for node', node_dict['name'], 'in', self.root_nodes)
                    graph_node = self.check_existing_node(node_dict['name'], self.root_nodes)
                else:
                    graph_node = self.check_existing_node(node_dict['name'], stack[-1].children)
                
                if graph_node is None:
                    frame = Frame({'name': node_dict['name']})
                    if node_dict['depth'] == 0:
                        graph_node = Node(frame, parent=None)
                        self.root_nodes.append(graph_node)
                    else:
                        graph_node = Node(frame, parent=stack[-1])
                        stack[-1].add_child(graph_node)
                
                    node_dict['node'] = graph_node
                    # node_dict['rank'] = rank_range
                    graph_node.frame.attrs['ranks'] = [rank_range]

                    values_dict = self.stats_dict[node_dict['name']]
                    for key in values_dict:
                        node_dict[key] = values_dict[key]
                    
                    dicts.append(node_dict)
                else:
                    # node already exists - let's updates rank ranges
                    self._update_rank_ranges(graph_node, rank_range)

                
                stack.append(graph_node)
                        
            # create df from dicts
            df = pd.DataFrame(dicts)
        
        return df
    
    def _update_rank_ranges(self, graph_node: Node, new_rank_range: (int, int)):
        # existing_ranges: List[(int, int)] = graph_node.frame.attrs.get('ranks')
        ranges = graph_node.frame.attrs.get('ranks')
        # print('merginf ranges', ranges, new_rank_range)
        # exit(0)
        if(ranges[-1][1] == new_rank_range[0]):
            ranges[-1] = (ranges[-1][0], new_rank_range[1])
        else:
            ranges.append(new_rank_range)
        graph_node.frame.attrs['ranks'] = ranges
        # new_ranges = []
        # for begin,end in sorted(ranges):
        #     if new_ranges and new_ranges[-1][1] >= begin - 1:
        #         new_ranges[-1][1] = max(new_ranges[-1][1], end)
        #     else:
        #         new_ranges.append([begin, end])
        # graph_node.frame.attrs['ranks'] = new_ranges



            
    def read_log_dir(self, log_dir: str):
        stat_file = os.path.join(log_dir, 'model_timing_stats')
        
        # log files are named model_timing.0, model_timing.1, etc.
        # we need to sort them by the number at the end of the file name
        log_files = os.path.join(log_dir, 'model_timing.*')
        log_files = sorted(glob.glob(log_files))
        # print(log_files)
        
        if len(log_files) == 0:
            raise ValueError('No log files found')
        
        # get the ranks postfix from all log files
        ranks = []
        offset = len('model_timing.') + log_files[0].rfind('model_timing.')
        for log_file in log_files:
            ranks.append(int(log_file[offset:]))
        # get total number of ranks from the stat file
        self.read_stat_file(stat_file)
        ranks.append(self.num_total_ranks)
        # print(ranks)

        self.root_nodes: List[Node] = []

        self.numeric_metric_names = ['Wallclock', 'max', 'min', 'UTR Overhead']

        dfs = []
        for i in range(len(log_files)):
            log_file = log_files[i]
            start_rank = ranks[i]
            end_rank = ranks[i + 1]
            # print(log_file)
            new_df = self.read_single_log(log_file, start_rank, end_rank)
            print('new df', new_df)
            dfs.append(new_df)
            # break
        df = pd.concat(dfs)
        df['rank_ranges'] = df['node'].apply(lambda x: x.frame.attrs['ranks'])
        df = df.set_index('node')
        graph = Graph(self.root_nodes)
        graph.enumerate_traverse()
        gf = GraphFrame(graph, df, [], self.numeric_metric_names, default_metric='Wallclock')
        return gf
                    
    # def get_num_total_ranks(self, stat_file) -> int:
    #     with open(stat_file, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             # Find the line that contains the number of ranks
    #             # '***** GLOBAL STATISTICS (  3456 MPI TASKS) *****'
    #             if line.startswith('***** GLOBAL STATISTICS'):
    #                 return int(line.split()[4])
    #     return 0
    
    def read_stat_file(self, stat_file):
        # print('reading stat file')
        with open(stat_file, 'r') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                # Find the line that contains the number of ranks
                # '***** GLOBAL STATISTICS (  3456 MPI TASKS) *****'
                if line.startswith('***** GLOBAL STATISTICS'):
                    self.num_total_ranks = line.split()[4]
                
                # we want to stop at line that looks like this
                # name                                            on  processes  threads        count      walltotal   wallmax (proc   thrd  )   wallmin (proc   thrd  )
                # Stats for thread 0:
                if line.startswith('name '):
                    metric_arr = line.strip().split()    
                    metric_arr = [elem.strip('(').strip(')') for elem in metric_arr if elem.strip('(').strip(')') != '']
                    # we're looking for specific metrics
                    # let's assert their positions
                    assert metric_arr[0] == 'name'
                    assert metric_arr[3] == 'threads'
                    assert metric_arr[4] == 'count'
                    assert metric_arr[5] == 'walltotal'
                    assert metric_arr[6] == 'wallmax'
                    assert metric_arr[9] == 'wallmin'
                    lines = lines[index + 1:]
                    break
            # set up arrays to make a df
            # names = []
            # threads = []
            # counts = []
            # walltotals = []
            # wallmaxs = []
            # wallmins = []
            self.stats_dict = {}

            for line in lines:
                line_arr = line.strip().split('"')
                if(len(line_arr) < 2):
                    continue
                name = line_arr[1]
                line_arr = line_arr[2].split()
                for elem in line_arr:
                    elem = elem.strip('(').strip(')')
                    # print(elem)
                line_arr = [elem.strip('(').strip(')') for elem in line_arr if elem.strip('(').strip(')') != '']
                
                threads = int(line_arr[2])
                count = int(float((line_arr[3])))
                walltotal = float(line_arr[4])
                wallmax = float(line_arr[5])
                wallmin = float(line_arr[8])
                self.stats_dict[name] = {'threads': threads, 'count': count, 'walltotal': walltotal, 'wallmax': wallmax, 'wallmin': wallmin}

                # names.append(name)
                # threads.append(line_arr[2])
                # counts.append(line_arr[3])
                # walltotals.append(line_arr[4])
                # wallmaxs.append(line_arr[5])
                # wallmins.append(line_arr[8])

                # # line_arr = [name] + line_arr
                # print(metric_arr)
                # print(line_arr)
                # break
            # self.stat_df = pd.DataFrame({'name': names, 'threads': threads, 'count': counts, 'walltotal': walltotals, 'wallmax': wallmaxs, 'wallmin': wallmins})
            # self.stat_df = self.stat_df.set_index('name')
            stat_df = pd.DataFrame(self.stats_dict,)
            # print(stat_df)
            # print(self.stat_df.head())
            # print(self.stat_df.loc['CPL:INIT']['threads'])

        return 0
                