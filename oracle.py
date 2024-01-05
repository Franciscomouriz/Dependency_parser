from data_dictionaries import deprel_dictionary

class Oracle:

    def __init__(self):
        self.sentence = None
        self.deprel_dict = deprel_dictionary
        self.states = []

    def start_oracle(self, tree):
        self.sentence = tree
        state = self.initial_state(self.sentence)
        
        while not self.final_state(state):

            if self.left_arc_is_valid(state) and self.left_arc_is_correct(state, self.sentence):
                state, new_stack, new_buffer = self.apply_left_arc(state, self.sentence)
            
            elif self.right_arc_is_valid(state) and self.right_arc_is_correct(state, self.sentence):
                state, new_stack, new_buffer = self.apply_right_arc(state, self.sentence)

            elif self.reduce_is_valid(state) and self.reduce_is_correct(state, self.sentence):
                state, new_stack, new_buffer = self.apply_reduce(state)

            else:
                state, new_stack, new_buffer = self.apply_shift(state)
            
            self.states.append(state)
            new_arcs = state['arcs'].copy()
            state = {'stack': new_stack, 'buffer': new_buffer, 'arcs': new_arcs, 'action': state['action']}

        last_state = {'stack': state['stack'], 'buffer': state['buffer'], 'arcs': state['arcs'], 'action': "END"}
        self.states.append(last_state)

        return self.finish_sentence()

    ################################################################################################
    ################################## INITIALIZATION ##############################################
    ################################################################################################
               
    def initiate_buffer(self, tree):
        buffer = []
        for row in tree.values:
            if row[0] != '0':
                buffer.append(row[0])
        return buffer
    
    def obtain_initial_stack(self, tree):
        return [tree.iloc[0][0]]
    
    def initial_state(self, sentence):
        stack = self.obtain_initial_stack(sentence)
        buffer = self.initiate_buffer(sentence)
        state = {'stack': stack, 'buffer': buffer, 'arcs': [], 'action': ""}
        return state
    
    def initial_state_predict(self, sentence, n_features = 2):
        state = self.initial_state(sentence)
        state['action'] = 'SHIFT'
        return state
    
    def final_state(self, state):
        return len(state['buffer']) == 0
    
    ################################################################################################
    ######################################## ENDING ################################################
    ################################################################################################

    def finish_sentence(self):
        states = self.states.copy()
        self.sentence = None
        self.states = []
        return states

    
    ##############################################################################################
    #################################### PRECONDITIONS ###########################################
    ##############################################################################################

    def check_in_tuple_list(self, arcs_list, arc):
        for arc_aux in arcs_list:
            if arc[0] != '_':
                if arc_aux[0] == arc[0] and arc_aux[2] == arc[2]:
                    return True
            else:
                if arc_aux[2] == arc[2] :
                    return True
        return False
    
    def left_arc_is_valid(self, state):
        last_stack = state['stack'][-1]
        first_buffer = state['buffer'][0]
        
        if last_stack is not '0' and not self.check_in_tuple_list(state['arcs'], ('_', '_',last_stack)):
            return True
        else:
            return False
        
    def right_arc_is_valid(self, state):
        last_stack = state['stack'][-1]
        first_buffer = state['buffer'][0]
        
        if not self.check_in_tuple_list(state['arcs'], ('_', '_', first_buffer)):
            return True
        else:
            return False
        
    def reduce_is_valid(self, state):
        last_stack = state['stack'][-1]
        if self.check_in_tuple_list(state['arcs'], ('_', '_',last_stack)):
            return True
        else:
            return False
        
    ##############################################################################################
    ##################################### VALIDATIONS ############################################
    ##############################################################################################

    def right_arc_is_correct(self, state, sentence):
        last_stack = state['stack'][-1]
        first_buffer = state['buffer'][0]
        # Locate first_buffer element in sentence['ID']
        first_buffer_row = sentence.loc[sentence['ID'] == first_buffer]

        if first_buffer_row['HEAD'].values[0] == last_stack:
            return True
        else:
            return False
        
    def left_arc_is_correct(self, state, sentence):
        last_stack = state['stack'][-1]
        first_buffer = state['buffer'][0]
        last_stack_row = sentence.loc[sentence['ID'] == last_stack]

        if last_stack_row['HEAD'].values[0] == first_buffer:
            return True
        else:
            return False
        
    def reduce_is_correct(self, state, sentence):
        last_stack = state['stack'][-1]
        rest_buffer = state['buffer'][1:]
        # If any element in rest_buffer has last_stack as head, return False
        for element in rest_buffer:
            element_row = sentence.loc[sentence['ID'] == element]
            if element_row['HEAD'].values[0] == last_stack:
                return False
        return True
    
    ##############################################################################################
    ####################################### ACTIONS ##############################################
    ##############################################################################################

    def apply_left_arc(self, state, sentence, label = None):
        last_stack = state['stack'][-1]
        first_buffer = state['buffer'][0]

        last_stack_row = sentence.loc[sentence['ID'] == last_stack]
        if label is None:
            label = last_stack_row['DEPREL'].values[0]

        new_arcs = state['arcs']
        new_arcs.append((first_buffer, label, last_stack))
        new_action = 'LEFT-ARC'
        new_state = {'stack': state['stack'], 'buffer': state['buffer'], 'arcs': new_arcs, 'action': new_action}

        new_stack = state['stack'][:-1]
        new_buffer = state['buffer']
        return new_state, new_stack, new_buffer

    def apply_right_arc(self, state, sentence, label = None):
        last_stack = state['stack'][-1]
        first_buffer = state['buffer'][0]

        first_buffer_row = sentence.loc[sentence['ID'] == first_buffer]
        if label is None:
            label = first_buffer_row['DEPREL'].values[0]

        new_arcs = state['arcs']
        new_arcs.append((last_stack, label, first_buffer))
        new_action = 'RIGHT-ARC'
        new_state = {'stack': state['stack'], 'buffer': state['buffer'], 'arcs': new_arcs, 'action': new_action}

        new_stack = state['stack'].copy()
        new_stack.append(first_buffer)
        new_buffer = state['buffer'][1:]
        return new_state, new_stack, new_buffer
    
    def apply_reduce(self, state):
        last_stack = state['stack'][-1]

        new_arcs = state['arcs']
        new_action = 'REDUCE'
        new_state = {'stack': state['stack'], 'buffer': state['buffer'], 'arcs': new_arcs, 'action': new_action}

        new_stack = state['stack'].copy()
        new_stack.remove(last_stack)
        new_buffer = state['buffer']
        return new_state, new_stack, new_buffer
    
    def apply_shift(self, state):
        first_buffer = state['buffer'][0]
        rest_buffer = state['buffer'][1:]

        new_arcs = state['arcs']
        new_action = 'SHIFT'
        new_state = {'stack': state['stack'], 'buffer': state['buffer'], 'arcs': new_arcs, 'action': new_action}

        new_stack = state['stack'].copy()
        new_stack.append(first_buffer)
        new_buffer = rest_buffer

        return new_state, new_stack, new_buffer

    ##############################################################################################
    ##################################### PREDICTION #############################################
    ##############################################################################################

    def is_valid_transition(self, state, transition):
        if transition == 'LEFT-ARC':
            return self.left_arc_is_valid(state)
        elif transition == 'RIGHT-ARC':
            return self.right_arc_is_valid(state)
        elif transition == 'REDUCE':
            return self.reduce_is_valid(state)
        elif transition == 'SHIFT':
            return True
        else:
            return False
        
    def apply_transition(self, state, transition, sentence, label):
        if transition == 'LEFT-ARC':
            state, new_stack, new_buffer = self.apply_left_arc(state, sentence, label)
        elif transition == 'RIGHT-ARC':
            state, new_stack, new_buffer = self.apply_right_arc(state, sentence, label)
        elif transition == 'REDUCE':
            state, new_stack, new_buffer = self.apply_reduce(state)
        elif transition == 'SHIFT':
            state, new_stack, new_buffer = self.apply_shift(state)

        new_arcs = state['arcs'].copy()
        state = {'stack': new_stack, 'buffer': new_buffer, 'arcs': new_arcs, 'action': state['action']}
        return state