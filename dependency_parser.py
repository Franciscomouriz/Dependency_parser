from process_data import ProcessData
from oracle import Oracle
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import copy
import os
import subprocess

class DependencyParser:

    def __init__(self, processData: ProcessData = None):
        self.processData = processData
        self.train_data = ()
        self.test_data = ()
        self.dev_data = ()
        self.tokenizer = None
        self.model = None

    def create_tokenizer(self, data_words):
        tokenizer = Tokenizer(filters="", oov_token="<unk>")
        tokenizer.fit_on_texts(data_words)
        self.tokenizer = tokenizer

    def prepare_data(self, n_features = 2):
        # Prepare samples with n_features
        x_train, y_train = self.prepare_samples_model(self.processData.train_data["samples"], self.processData.train_data["dataframes"], n_features)
        x_test, y_test = self.prepare_samples_model(self.processData.test_data["samples"], self.processData.test_data["dataframes"], n_features)
        x_dev, y_dev = self.prepare_samples_model(self.processData.dev_data["samples"], self.processData.dev_data["dataframes"], n_features)

        # Prepare samples with n_features
        x_train_tokenized = self.tokenize_X_data(self.tokenizer, x_train)
        x_train_tagged = self.X_to_tags(x_train, n_features)
        X_train_list = self.create_list_inputs(x_train_tokenized, x_train_tagged, n_features)

        x_test_tokenized = self.tokenize_X_data(self.tokenizer, x_test)
        x_test_tagged = self.X_to_tags(x_test, n_features)
        X_test_list = self.create_list_inputs(x_test_tokenized, x_test_tagged, n_features)

        x_dev_tokenized = self.tokenize_X_data(self.tokenizer, x_dev)
        x_dev_tagged = self.X_to_tags(x_dev, n_features)
        X_dev_list = self.create_list_inputs(x_dev_tokenized, x_dev_tagged, n_features)


        # Prepare Y data
        y_train_tagged = self.y_to_tags(y_train)
        y_train_categorical = self.y_to_categorical(y_train_tagged)
        y_test_tagged = self.y_to_tags(y_test)
        y_test_categorical = self.y_to_categorical(y_test_tagged)
        y_dev_tagged = self.y_to_tags(y_dev)
        y_dev_categorical = self.y_to_categorical(y_dev_tagged)

            
        # Save the model samples
        self.train_data = [np.array(X_train_list), y_train_categorical]
        self.test_data = [np.array(X_test_list), y_test_categorical]
        self.dev_data = [np.array(X_dev_list), y_dev_categorical]

    def create_and_fit_model(self, epochs = 10, batch_size = 64, n_features = 2):
        # Input layer with 4 inputs
        inputs = tf.keras.Input(shape=(4*n_features,))
        # Embedding layer
        x = tf.keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=128, mask_zero=True)(inputs)
        # LSTM layer
        x = tf.keras.layers.LSTM(128, return_sequences=False)(x)
        # Output layer
        output1 = tf.keras.layers.Dense(6, activation="softmax", name="output1")(x)
        output2 = tf.keras.layers.Dense(72, activation="softmax", name="output2")(x)
        # Create the model
        model = tf.keras.Model(inputs, [output1, output2])
        # Print the model summary
        model.summary()
        # Compile the model
        model.compile(loss={'output1': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'}, optimizer="adam", metrics=["accuracy"])
        # Fit the model
        model.fit(self.train_data[0], {'output1': self.train_data[1][0], 'output2': self.train_data[1][1]}, epochs=epochs, batch_size=batch_size,validation_data=self.dev_data)
        # Save the model
        self.model = model
        #Evaluate the model
        print("\n\nEvaluate on test data")
        results = self.evaluate_model()
        print("test loss, test acc:", results)

    def evaluate_model(self):
        results = self.model.evaluate(self.test_data[0], {'output1': self.test_data[1][0], 'output2': self.test_data[1][1]})
        print("test loss, test acc:", results)
        return results

    def predict(self, sentences, n_features = 2):
        print("\n\nEvaluate on dev data")
        print("Predicting...")
        oracle = Oracle()
        # Obtain initial state of all sentences
        states = {}
        dataframes = {}
        for i, sentence in enumerate(sentences):
            states[i] = oracle.initial_state_predict(sentence, n_features)
            dataframes[i] = sentence

        # Predict the actions until all sentences are finished
        completed_trees = {}
        while states != {}:
            # Obtain batch of features
            test_data = self.prepare_test_prediction_data(states, dataframes, n_features=n_features)
            # Predict the action
            pred_arcs, pred_deprel = self.model.predict(test_data[0])
            # Apply the action
            for i, key in enumerate(list(states.keys())):
                prediction_arc = pred_arcs[i]
                prediction_label = pred_deprel[i]

                valid = False
                while not valid:
                    # Obtain the predictions
                    predicted_arc = np.argmax(prediction_arc)
                    predicted_label = np.argmax(prediction_label)
                    # Put the predicted arc as 0 to avoid predicting it again
                    prediction_arc[predicted_arc] = 0
                    # Decode the predictions
                    predicted_arc = self.translate_tags(predicted_arc, "arc")
                    predicted_label = self.translate_tags(predicted_label, "deprel")
                    # Verify if the arc is valid
                    valid = oracle.is_valid_transition(states[key],predicted_arc)
                # Create the transition with the predicted arc
                states[key] = oracle.apply_transition(states[key], predicted_arc, sentences[i], predicted_label)

            # Delete the finished sentences
            finished_states = []
            for i, state in states.items():
                if oracle.final_state(states[i]):
                    finished_states.append(i)
                    states[i] = last_state = {'stack': states[i]['stack'], 'buffer': states[i]['buffer'], 'arcs': states[i]['arcs'], 'action': "END"}
                    completed_trees[i] = states[i]

            for i in finished_states:
                del states[i]
                del dataframes[i]

        # Correct the predicted results
        completed_trees = self.correct_predicted_results(completed_trees)

        return completed_trees
    
    def conllu_evaluation(self, predictions, n_features):
        copy_data_test = copy.deepcopy(self.processData.test_data["dataframes"])
        copy_completed_trees = copy.deepcopy(predictions)

        # Delete ID 0 from the dataframe
        for index, dataframe in enumerate(copy_data_test):
            dataframe.drop(0, inplace=True)

        # Iterate at the same time over the copy_data_test and the completed_trees
        for index, dataframe in enumerate(copy_data_test):
            arcs = sorted(copy_completed_trees[index]['arcs'], key = lambda x: int(x[2]))
            # Substitute the HEAD and DEPREL columns by the values (_, HEAD, DEPREL) of the arcs list
            dataframe["HEAD"] = [arc[0] for arc in arcs]
            dataframe["DEPREL"] = [arc[1] for arc in arcs]

        # Convert the dataframes to conllu format
        os.makedirs(f"evaluation/{n_features}_features", exist_ok=True)
        self.convert_predictions_to_conllu(copy_data_test, "predicted_trees", n_features)
        self.convert_predictions_to_conllu(self.processData.test_data["dataframes"], "original_trees", n_features)

        # Evaluate the results
        # If evaluation/conll18_ud_eval.py doesn't exist, download it
        if not os.path.exists("evaluation/conll18_ud_eval.py"):
            os.makedirs("evaluation", exist_ok=True)
            os.system("wget -O evaluation/conll18_ud_eval.py https://universaldependencies.org/conll18/conll18_ud_eval.py")

        with open(f"evaluation/{n_features}_features/results.txt", "w") as outfile:
            subprocess.run(["python3", f"evaluation/conll18_ud_eval.py", f"evaluation/{n_features}_features/original_trees.conllu", f"evaluation/{n_features}_features/predicted_trees.conllu"], stdout=outfile)
            subprocess.run(["python3", f"evaluation/conll18_ud_eval.py", f"evaluation/{n_features}_features/original_trees.conllu", f"evaluation/{n_features}_features/predicted_trees.conllu", "-v"], stdout=outfile)

            subprocess.Popen(["cat", f"evaluation/{n_features}_features/results.txt"])



    ##########################################################################################
    ################################ Processing samples ######################################
    ##########################################################################################

    def prepare_samples_model(self, samples, dataframes, n_features = 2):

        if not isinstance(samples, list):
            samples = [samples]
        if not isinstance(dataframes, list):
            dataframes = [dataframes]

        model_samples = []
        # Iterar a la vez sobre samples y dataframes
        for sentence, dataframe in zip(samples, dataframes):
            if not isinstance(sentence, list):
                sentence = [sentence]
            # Iterar sobre cada elemento de sentence y dataframe
            for sampleS in sentence:
                # From data["stack"] select the n_features last elements
                stack = self.extract_stack_features(sampleS["stack"], n_features, dataframe)
                # From data["buffer"] select the n_features first elements
                buffer = self.extract_buffer_features(sampleS["buffer"], n_features, dataframe)
                stack_upos = self.extract_stack_upos(sampleS["stack"], n_features, dataframe)
                buffer_upos = self.extract_buffer_upos(sampleS["buffer"], n_features, dataframe)
                # Obtain the action
                action = sampleS["action"]
                # If action is SHIFT or REDUCE, arc is None
                if action == "SHIFT" or action == "REDUCE":
                    arc = "None"
                else:
                    arc = sampleS["arcs"][-1][1] if sampleS["arcs"] else "None"

                # Create the model sample
                model_sample = (stack, buffer, stack_upos, buffer_upos, action, arc)
                model_samples.append(model_sample)
        x_data = [model_sample[:4] for model_sample in model_samples]
        y_data = [model_sample[4:] for model_sample in model_samples]
        return x_data, y_data

    def extract_stack_features(self, stack, n_features, dataframes):
        # Obtain the n_features last elements of the stack
        stack_elements = stack[-n_features:]
        stack_features = []
        if len(stack_elements) < n_features:
            for i in range(n_features - len(stack_elements)):
                stack_features.append("MASK")
        for feature in stack_elements:
            feature = int(feature)
            stack_features.append(dataframes.loc[feature]["FORM"])

        return stack_features
    
    def extract_buffer_features(self, buffer, n_features, dataframes):
        # Obtain the n_features first elements of the buffer
        buffer_elements = buffer[:n_features]
        buffer_features = []
        for feature in buffer_elements:
            feature = int(feature)
            buffer_features.append(dataframes.loc[feature]["FORM"])
        if len(buffer_elements) < n_features:
            for i in range(n_features - len(buffer_elements)):
                buffer_features.append("MASK")

        return buffer_features
    
    def extract_stack_upos(self, stack, n_features, dataframes):
        # Obtain the n_features last elements of the stack
        stack_elements = stack[-n_features:]
        stack_features = []
        if len(stack_elements) < n_features:
            for i in range(n_features - len(stack_elements)):
                stack_features.append("MASK")
        for feature in stack_elements:
            feature = int(feature)
            stack_features.append(dataframes.loc[feature]["UPOS"])

        return stack_features
    
    def extract_buffer_upos(self, buffer, n_features, dataframes):
        # Obtain the n_features first elements of the buffer
        buffer_elements = buffer[:n_features]
        buffer_features = []
        for feature in buffer_elements:
            feature = int(feature)
            buffer_features.append(dataframes.loc[feature]["UPOS"])
        if len(buffer_elements) < n_features:
            for i in range(n_features - len(buffer_elements)):
                buffer_features.append("MASK")

        return buffer_features
    
    ##########################################################################################
    ################################### Prepare X data #######################################
    ##########################################################################################

    def tokenize_X_data(self, tokenizer, data):
        new_data = []
        for element in data:
            # Element has format [[Stack1,Stack2], [Buffer1,Buffer2]]
            # Stack and Buffer are lists of words
            # Tokenize Stack and Buffer
            elements_tokenized = []
            elements_tokenized.append(self.tokenize_data(tokenizer, element[0]))
            elements_tokenized.append(self.tokenize_data(tokenizer, element[1]))
            new_data.append(elements_tokenized)
        return new_data
    
    def create_list_inputs(self, data, data_upos, nfeatures):
        total_list = []
        for row, row_upos in zip(data, data_upos):
            # Create a list with format [[1, 2, 3, 4], [5, 6, 7, 8], ...]
            list = []
            aux = []
            for column, column_upos in zip(row, row_upos):
                for i in range(nfeatures):
                    list.append(column[i][0])
                    aux.append(column_upos[i])
            list.extend(aux)
            total_list.append(list)
        return total_list    
    
    ##########################################################################################
    ################################### Prepare y data #######################################
    ##########################################################################################

    def y_to_tags(self, y_data):
        y_data_categorical = []
        deprel_dictionary = self.processData.deprel_dict
        arcs_dictionary = self.processData.arcs_dict
        #print(y_data)
        for element in y_data:
            elements_categorical = []
            elements_categorical.append(arcs_dictionary[element[0]])
            elements_categorical.append(deprel_dictionary[element[1]])
            y_data_categorical.append(elements_categorical)
        return y_data_categorical
    
    def y_to_categorical(self, y_data):
        output1_labels, output2_labels = zip(*y_data)
        output1_labels = tf.keras.utils.to_categorical(output1_labels, num_classes=6)
        output2_labels = tf.keras.utils.to_categorical(output2_labels, num_classes=72)
        return [output1_labels, output2_labels]
    
    def X_to_tags(self, x_data, n_features = 2):
            x_data_categorical = []
            upos_dictionary = self.processData.upos_dict
            for element in x_data:
                #print(element)
                elements_categorical = []
                elements_1 = []
                elements_2 = []
                for i in range(n_features):
                    elements_1.append(upos_dictionary[element[2][i]])
                    elements_2.append(upos_dictionary[element[3][i]])
                elements_categorical.append(elements_1)
                elements_categorical.append(elements_2)
                x_data_categorical.append(elements_categorical)
            return x_data_categorical

    
    def x_to_categorical(self, x_data):
        #print(x_data)
        stack_1, buffer_1, stack_2, buffer_2 = zip(*x_data)
        stack_1 = tf.keras.utils.to_categorical(stack_1, num_classes=20)
        stack_2 = tf.keras.utils.to_categorical(stack_2, num_classes=20)
        buffer_1 = tf.keras.utils.to_categorical(buffer_1, num_classes=20)
        buffer_2 = tf.keras.utils.to_categorical(buffer_2, num_classes=20)
        return [stack_1, buffer_1, stack_2, buffer_2]
    

    ##########################################################################################
    ################################## Prepare test data ######################################
    ##########################################################################################

    def prepare_test_data(self, n_features = 2):
        x_test, y_test = self.prepare_samples_model(self.processData.test_data["samples"], self.processData.test_data["dataframes"], n_features)

        x_test_tokenized = self.tokenize_X_data(self.tokenizer, x_test)
        x_test_tagged = self.X_to_tags(x_test, n_features)
        X_test_list = self.create_list_inputs(x_test_tokenized, x_test_tagged, n_features)

        y_test_tagged = self.y_to_tags(y_test)
        y_test_categorical = self.y_to_categorical(y_test_tagged)

        self.test_data = [np.array(X_test_list), y_test_categorical]


    def prepare_test_prediction_data(self, states, dataframes, n_features = 2):
        # Obtain batch of features
        samples = {}
        for i, state in states.items():
            x_data, y_data = self.prepare_samples_model(state, dataframes[i], n_features = n_features)
            samples[i] = (x_data, y_data)

            # Prepare the data for the model
            x_data = []
            y_data = []
            for i, sample in samples.items():
                x_data += sample[0]
                y_data += sample[1]

            x_data_tokenized = self.tokenize_X_data(self.tokenizer, x_data)
            x_data_tagged = self.X_to_tags(x_data, n_features)
            x_data_list = self.create_list_inputs(x_data_tokenized, x_data_tagged, n_features)

            y_data_tagged = self.y_to_tags(y_data)
            y_data_categorical = self.y_to_categorical(y_data_tagged)

            test_data = [np.array(x_data_list), y_data_categorical]
        return test_data

    ##########################################################################################
    ################################ Tokenizer functions #####################################
    ##########################################################################################

    def load_tokenizer(self, path):
        with open(path, "rb") as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer

    def save_tokenizer(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def tokenize_data(self, tokenizer, data):
        # Tokenize the data
        tokenized_data = tokenizer.texts_to_sequences(data)
        return tokenized_data

    ##########################################################################################
    ################################## Model functions #######################################
    ##########################################################################################

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def save_model(self, path):
        self.model.save(path)
        
    ##########################################################################################
    ########################### Translate evaluated results ##################################
    ##########################################################################################

    def obtain_tags(self,predictions):
        tags = []
        for sentence in predictions:
            tags.append(np.argmax(sentence))
        return tags
            
    def translate_tags(self, tags, type_tags):
        # Selección del diccionario correspondiente
        if type_tags == "arc":
            samples = self.processData.arcs_dict
        elif type_tags == "deprel":
            samples = self.processData.deprel_dict
        else:
            raise ValueError("type_tags must be 'arc' or 'deprel'")
        
        samples = {v: k for k, v in samples.items()}

        # Asegurarse de que tags sea una lista
        if not isinstance(tags, list):
            tags = [tags]

        # Traducción de los tags
        result = [samples[tag] for tag in tags]

        if len(result) == 1:
            return result[0]
        
        return result

    ##########################################################################################
    ############################# Correct predicted results ##################################
    ##########################################################################################

    def correct_predicted_results(self, predictions):
        oracle = Oracle()
        # Correct trees without Root
        trees_no_root = {tree: data for tree, data in predictions.items() 
                    if not any(arc[0] == '0' for arc in data["arcs"])}
        
        
        for tree in trees_no_root:
            words = oracle.initiate_buffer(self.processData.test_data["dataframes"][tree])
            words_headed = [arc[2] for arc in trees_no_root[tree]["arcs"]]
            words_non_headed = [word for word in words if word not in words_headed]
            
            # Create an arc from the root to the first word in words_non_headed
            arc = ('0', "_", words_non_headed[0])
            trees_no_root[tree]["arcs"].append(arc)

        # Correct trees with various roots
        various_roots_trees = {tree: data for tree, data in predictions.items()
                       if sum(arc[0] == '0' for arc in data["arcs"]) > 1}


        for tree in various_roots_trees:
            arcs_with_root = [arc for arc in various_roots_trees[tree]["arcs"] if arc[0] == '0']
            new_arcs = [arcs_with_root[0]]
            new_root = arcs_with_root[0][2]
            for arc in arcs_with_root[1:]:
                new_label = self.processData.test_data["dataframes"][tree]["DEPREL"][int(arc[2])]
                new_arcs.append((new_root, new_label, arc[2]))

            # Delete arcs with root from the list of arcs
            old_arcs = [arc for arc in various_roots_trees[tree]["arcs"] if arc[0] != '0']
            for arc in old_arcs:
                new_arcs.append(arc)

            various_roots_trees[tree]["arcs"] = new_arcs

        # Correct trees with no parent
        incomplete_trees = {tree: data for tree, data in predictions.items()
                    if set(oracle.initiate_buffer(self.processData.test_data["dataframes"][tree])) - {arc[2] for arc in data["arcs"]}}
        
        
        for tree in incomplete_trees:
            words = oracle.initiate_buffer(self.processData.test_data["dataframes"][tree])
            for arc in incomplete_trees[tree]["arcs"]:
                if arc[0] == '0':
                    parent = arc[2]
                    break
            for word in words:
                if word not in [arc[2] for arc in incomplete_trees[tree]["arcs"]]:
                    arc = (parent, "_", word)
                    incomplete_trees[tree]["arcs"].append(arc)
        
        return predictions


    ##########################################################################################
    ################################ Convert to conllu file ##################################
    ##########################################################################################

    def convert_predictions_to_conllu(self, predictions, filename, n_features):
        conllu_file = ""
        for index, dataframe in enumerate(predictions):
            conllu_file += "# sent_id = " + str(index) + "\n"
            conllu_file += "# text = " + " ".join(dataframe["FORM"]) + "\n"

            for index, row in dataframe.iterrows():
                if index != 0:
                    line = "\t".join(str(item) for item in row if pd.notna(item))
                    conllu_file += line

            conllu_file += "\n"  # Añadir una línea en blanco al final de cada entrada
        with open(f"evaluation/{n_features}_features/{filename}.conllu", "w") as f:
            f.write(conllu_file)
