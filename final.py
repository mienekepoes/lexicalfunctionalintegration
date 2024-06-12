#==================================================================#
# Kat Van der Poorten                                              #  
#                                                                  #       
# Modeling the Evolutionary Transition from Holistic Signaling to  #
# Compositional Language through Lexical-Functional Integration    #
#                                                                  # 
# Thesis presented in the fullfilment of the requirements for      #
# the degree of Master of Science in Biology                       #
#                                                                  #
# Academic year 2023-2024                                          #
#                                                                  #       
# Supervisor: Prof. P. van den Berg (KU Leuven)                    #
# Co-supervisor: Prof. S. Miyagawa (MIT)                           #
#==================================================================#


# Imports and seeds

import os
import sys
import re # Regular expressions for splitting integrated meaning into constituents
import itertools
import random # Random if you need to generate over heterogeneous lists (np cannot do this)
import numpy as np
import pandas as pd
import copy
from collections import Counter
import matplotlib.pyplot as plt



#============#
# Parameters #
#============#

### SIMULATION PARAMETERS ###

simulations = int(sys.argv[1])


num_generations = 1000
population_size = 100
max_population_size = 150

mutation_rate = 0.01

base_association_strength = 5 # The base association strength of item in lexicon
                              # i.e. an agent can use a new item 5x without success before
                              # it disappeaers from the lexicon

transparency = 0.5 # Chance that a meaning is transparent (S and O cannot be confused)

# Initial proportions of agent types
proportion_holistic_memorizers = 0.5
proportion_pattern_fragmenters = 0.5

communication_rounds = 2000 # The number of communication rounds per generation
cultural_evolution_rate = 0.3 
max_meanings = 500 # The maximum number of meanings in the semantic space

cue_availability = int(sys.argv[2])
cue_reliability = 1
environmental_change_rate = 0.1

### AGENT PARAMETERS ###

buffer_size = 7 # Size of short term memory 
max_age = 35
reproductive_age = 15 # Needed because of overlapping generations

cognitive_cost = 100 # The cognitive cost pattern fragmenters pay
lexical_cost = 1 # The cost associated with growing lexicon

cultural_inheritance_rate = 0.75 # % of lexicon offspring inherits from parent (vertical cultural transmission)


#==================================#
# Language variables and functions #
#==================================#

### SYLLABIC SPACE ###

# The syllabic space of the language (50 syllables)
syllables = ["ra", "re", "ru", "ri", "ro",
             "ka", "ke", "ku", "ki", "ko",
             "ta", "te", "tu", "ti", "to",
             "na", "ne", "nu", "ni", "no",
             "ma", "me", "mu", "mi", "mo",
             "pa", "pe", "pu", "pi", "po",
             "ya", "yo", "yu", "yi", "yo",
             "la", "le", "lu", "li", "lo",
             "va", "ve", "vu", "vi", "vo"]

### UTTERANCE GENERATION FUNCTIONS ###

def generate_holistic_utterance(syllables, length):
    if length is None:
        length = np.random.randint(3, 7) # Generate a random length between 3 and 6
    return "".join(np.random.choice(syllables) for _ in range(length))

def generate_compositional_utterance(syllables, length):
    if length is None:
        length = np.random.randint(1, 3) # Generate a random length between 1 and 2
    return "".join(np.random.choice(syllables) for _ in range(length))

# Separate functions needed so that there is no way for the agents to detect the type of
# MUM that is being generated based on the length of the utterance. The length of the 
# utterances generated here has no influence on how the pattern fragmenters will fragment
# the utterance into its constituent parts.

### SEMANTIC SPACE ###

# The semantic space of the language containing all valid integrated meanings
class SemanticSpace:

    # The semantic space is the space of all possible integrated meanings that can be 
    # expressed in the language. This space represents the world-view of the agents, the
    # set of all possible events that can be referred to in the language. Meanings that 
    # can be expressed are biological relevant meanings, since the selection pressure
    # is assumed to be effective communication about the environment. 

    def __init__(self):
        self.content = []
        self.verbs = []
        self.subjs = []
        self.objs = []
        self.max_meanings = max_meanings

    def print_SP(self):
        print(self.content)

    def initialize_semantic_space(self):
        self.verbs = ['V' + str(i) for i in range(1, 4)]
        self.subjs = ['S' + str(i) for i in range(1, 4)]
        self.objs = ['O' + str(i) for i in range(1, 4)]
        self.content = [''.join(i) for i in itertools.product(self.verbs, self.subjs, self.objs) if random.random() <= 0.4]
                                        # The probability that a given combination of verb,
                                        # subject, and object is deemed biologically relevant
                                        # and added to the semantic space. The choice of 0.4
                                        # is arbitrary, but it is assumed that not all possible
                                        # combinations are biologically relevant. 
                                        # (This was also a trick for when the semantic space grows not letting it grow too fast 
                                        # - computational issue otherwise)

    # Due to cultural evolution (innovations in society) the semantic space needs to be able
    # to grow, to allow for new integrated meanings to be added. This is done by randomly adding new
    # verbs, objects, or subjects to the semantic space, and generating a new subset of valid
    # integrated meanings. 

    def grow_semantic_space(self):
        choice = random.choice([1, 2, 3])
        if choice == 1:
            new_verb = 'V' + str(len(self.verbs)+1)
            self.verbs.append(new_verb)
        elif choice == 2:
            new_subj = 'S' + str(len(self.subjs)+1)
            self.subjs.append(new_subj)
        else:
            new_obj = 'O' + str(len(self.objs)+1)
            self.objs.append(new_obj)

        new_meanings = [''.join(i) for i in itertools.product(self.verbs, self.subjs, self.objs) if random.random() <= 0.2]
        num_new_meanings = min(self.max_meanings - len(self.content), len(new_meanings))
        new_meanings_to_add = random.sample(new_meanings, num_new_meanings)
        self.content.extend(new_meanings_to_add)
                                  # The probability that a given combination with the new constituent
                                  # is deemed biologically relevant and added to the semantic space. 
                                  # This probability is lower than the one used during initialization,
                                  # to allow for more diversity in the semantic space, and also to prevent
                                  # too rapid expansion. It is also realistic that not all new cultural 
                                  # inventions will generate biologically relevant meanings that can 
                                  # combine with already existing constituents.

    def environment_changes(self):
        # The environment changes by removing a subset of the meanings from the semantic space
        # This allows for the agents to adapt to new circumstances and to prevent the agents 
        # from overfitting to the environment
        num_removals = int(len(self.content) * 0.05)
        removed_meanings = random.sample(self.content, num_removals)
        self.content = [meaning for meaning in self.content if meaning not in removed_meanings]

# Function to split the integrated meanings into constituents
def split_integrated_meaning(integrated_meaning):
    pattern = r'(\D+)(\d+)'
    matches = re.findall(pattern, integrated_meaning)
    verb = str(matches[0][0] + matches[0][1])
    subj = str(matches[1][0] + matches[1][1])
    obj = str(matches[2][0] + matches[2][1])
    return verb, subj, obj

### LEXICON ###

# The initial lexicon of all the agents, mapping holistic utterances to integrated meanings

def initialize_lexicon(semantic_space):
    lexicon = {}
    for meaning in semantic_space.content:
        lexicon[meaning] = {(generate_holistic_utterance(syllables, None)): (base_association_strength, 'HOLISTIC')}
    return lexicon

def print_lexicon(lexicon):
    data = []
    for meaning in lexicon:
        entries = lexicon[meaning]
        for entry in entries:
            data.append([meaning, entry, entries[entry][0], entries[entry][1]])
    
    df = pd.DataFrame(data, columns=['Meaning', 'Utterance', 'Association Strength', 'Type'])
    print(df, file=sys.stdout, flush=True)

def remove_homonyms_and_synonyms(lexicon):
    # Create a dictionary to store the counts of each meaning
    meaning_counts = {}

    # Loop through the lexicon and count the number of entries for each meaning
    for meaning, utterances in lexicon.items():
        if meaning in meaning_counts:
            meaning_counts[meaning] += len(utterances)
        else:
            meaning_counts[meaning] = len(utterances)

    # Create a list of meanings to remove
    meanings_to_remove = []

    # Loop through the meaning counts and remove meanings with more than 2 entries
    for meaning, count in meaning_counts.items():
        if count > 2:
            meanings_to_remove.append(meaning)
        else:
            # Loop through the utterances and count the number of entries for each one
            utterance_counts = {}
            for utterance, association_strength in utterances.items():
                if utterance in utterance_counts:
                    utterance_counts[utterance] += 1
                else:
                    utterance_counts[utterance] = 1

            # Create a list of utterances to remove
            utterances_to_remove = []
            for utterance, count in utterance_counts.items():
                if count > 2:
                    utterances_to_remove.append(utterance)

            # Remove the utterances from the lexicon
            for utterance in utterances_to_remove:
                del lexicon[meaning][utterance]

            # If there are no utterances left for the meaning, add it to the list of meanings to remove
            if not lexicon[meaning]:
                meanings_to_remove.append(meaning)

    # Remove the meanings from the lexicon
    for meaning in meanings_to_remove:
        del lexicon[meaning]

def count_shared_MUMs_PFs(population):
    # Get the lexicons of the agents
    lexicons = [agent.lexicon for agent in population if agent.type == 'Pattern Fragmenter']
    # For each lexicon, only retain the meaning and utterance and lose the association strength and meaning type
    lexicons = [{meaning: utterance for meaning, utterances in lexicon.items() for utterance in utterances} for lexicon in lexicons]
    # Count the amount of shared MUMs
    shared_MUMs = len(set.intersection(*map(set, lexicons)))
    return shared_MUMs

def count_shared_MUMs_HMs(population):
    # Get the lexicons of the agents
    lexicons = [agent.lexicon for agent in population if agent.type == 'Holistic Memorizer']
    # For each lexicon, only retain the meaning and utterance and lose the association strength and meaning type
    lexicons = [{meaning: utterance for meaning, utterances in lexicon.items() for utterance in utterances} for lexicon in lexicons]
    # Count the amount of shared MUMs
    shared_MUMs = len(set.intersection(*map(set, lexicons)))
    return shared_MUMs

def count_shared_compositional_MUMs(population):
    # Get the lexicons of the agents
    lexicons = [agent.lexicon for agent in population]
    # For each lexicon, only retain the meaning and utterance and lose the association strength and meaning type
    lexicons = [{meaning: utterance for meaning, utterances in lexicon.items() for utterance in utterances if utterances[utterance][1] == 'COMPOSITIONAL'} for lexicon in lexicons]
    # Count the amount of shared MUMs
    shared_MUMs = len(set.intersection(*map(set, lexicons)))
    return shared_MUMs


#========#
# Agents #
#========#

### HOLISTIC MEMORIZER ###

class HolisticMemorizer:

    def __init__(self, initial_lexicon):
        self.type = 'Holistic Memorizer'
        self.communicated_meanings = set()
        self.lexicon = copy.deepcopy(initial_lexicon)
        self.buffer = {}
        self.buffer_size = buffer_size 
        self.holistic_rules = 0
        self.compositional_rules = 0
        self.successful_interactions = 0
        self.fitness = 1
        self.age = 0

    ### HELPER FUNCTIONS ###
    def print_buffer(self):
        for meaning in self.buffer:
            print(f"{meaning}: {self.buffer[meaning]}", file=sys.stdout, flush=True)

    ### LEARNING ###
    def learn(self):

        # Check if the buffer is full
        if len(self.buffer) >= self.buffer_size:

            # Get the most frequent MUM from the buffer
            max_value = max([max(value.values()) for value in self.buffer.values()])
            retrieved_meanings = [key for key, value in self.buffer.items() if max(value.values()) == max_value]
            retrieved_meaning = random.choice(retrieved_meanings)
            retrieved_utterance = [key for key, value in self.buffer[retrieved_meaning].items() if value == max_value][0]

            # Check if the retrieved MUM is in the lexicon
            if retrieved_meaning in self.lexicon:
                if retrieved_utterance in self.lexicon[retrieved_meaning]:
                    self.lexicon[retrieved_meaning][retrieved_utterance] = (self.lexicon[retrieved_meaning][retrieved_utterance][0] + 1, 'HOLISTIC')
                else:
                    self.lexicon[retrieved_meaning][retrieved_utterance] = (base_association_strength, 'HOLISTIC')

            # Clear the buffer so the agent can learn from new experiences
            self.buffer = {}

    ### COMMUNICATION ###
    def send(self, receiver, semantic_space):

        # Select a meaning from the semantic space to communicate
        selected_meaning = str(np.random.choice(list(semantic_space.content)))
        # Add the selected meaning to the set of communicated meanings
        self.communicated_meanings.add(selected_meaning)

        # Check if the selected meaning is in the lexicon of the sender
        if selected_meaning in self.lexicon:
            if self.lexicon[selected_meaning]:
                # Retrieve the MUM with the highest association strength
                retrieved_utterance = [max(self.lexicon[selected_meaning], key=self.lexicon[selected_meaning].get)]
            else:
                retrieved_utterance = [generate_holistic_utterance(syllables, None)]
                self.lexicon[selected_meaning] = {retrieved_utterance[0]: (base_association_strength, 'HOLISTIC')}
        else:
            retrieved_utterance = [generate_holistic_utterance(syllables, None)]
            self.lexicon[selected_meaning] = {retrieved_utterance[0]: (base_association_strength, 'HOLISTIC')}

        # Communicate the MUM to the receiver (communication is the same for both receiver types)

        # Check if the MUM is in the lexicon of the receiver
        if selected_meaning in receiver.lexicon:
            if retrieved_utterance[0] in receiver.lexicon[selected_meaning]:
                # Strengthen the association strength of the MUM in the lexicon of both agents
                receiver.lexicon[selected_meaning][retrieved_utterance[0]] = (receiver.lexicon[selected_meaning][retrieved_utterance[0]][0] + 1, 'HOLISTIC')
                self.lexicon[selected_meaning][retrieved_utterance[0]] = (self.lexicon[selected_meaning][retrieved_utterance[0]][0] + 1, 'HOLISTIC')
                # Update communication success for both agents
                receiver.successful_interactions += 1
                self.successful_interactions += 1
            else: # The meaning is in the lexicon, but with a different utterance
                # Check for cues in the environment
                if np.random.rand() < cue_availability:
                    if np.random.rand() < cue_reliability:
                        # The meaning can be inferred from the context
                        # Add the meaning to the lexicon of the receiver
                        receiver.lexicon[selected_meaning] = {retrieved_utterance[0]: (base_association_strength, 'HOLISTIC')}
                        # Strengthen the MUM in the lexicon of the sender
                        self.lexicon[selected_meaning][retrieved_utterance[0]] = (self.lexicon[selected_meaning][retrieved_utterance[0]][0] + 1, 'HOLISTIC')
                        # Update communication success for both agents
                        self.successful_interactions += 1
                        receiver.successful_interactions += 1
                else: # There is no reliable cue available
                    # Add the MUM to the buffer of the receiver
                    if selected_meaning not in receiver.buffer:
                        receiver.buffer[selected_meaning] = {retrieved_utterance[0]: 1}
                    else:
                        receiver.buffer[selected_meaning][retrieved_utterance[0]] = (receiver.buffer[selected_meaning].get(retrieved_utterance[0], 0) + 1)
        else: # The meaning is not in the lexicon of the receiver
            # Check for cues
            if np.random.rand() < cue_availability:
                if np.random.rand() < cue_reliability:
                    # The meaning can be inferred from the context
                    # Add the meaning to the lexicon of the receiver
                    receiver.lexicon[selected_meaning] = {retrieved_utterance[0]: (base_association_strength, 'HOLISTIC')}
                    # Strengthen the MUM in the lexicon of the sender
                    self.lexicon[selected_meaning][retrieved_utterance[0]] = (self.lexicon[selected_meaning][retrieved_utterance[0]][0] + 1, 'HOLISTIC')
                    # Update communication success for both agents
                    self.successful_interactions += 1
                    receiver.successful_interactions += 1
            else: # There is no reliable cue available
                # Add the MUM to the buffer of the receiver
                if selected_meaning not in receiver.buffer:
                    receiver.buffer[selected_meaning] = {retrieved_utterance[0]: 1}
                else:
                    receiver.buffer[selected_meaning][retrieved_utterance[0]] = (receiver.buffer[selected_meaning].get(retrieved_utterance[0], 0) + 1)

        # Learning and updating
        self.learn()
        receiver.learn()

        # Remove MUMs with association strength 0 from the lexicon
        self.lexicon = {meaning: {utterance: (value, m_type) for utterance, (value, m_type) in self.lexicon[meaning].items() if value > 0} for meaning in self.lexicon}
        receiver.lexicon = {meaning: {utterance: (value, m_type) for utterance, (value, m_type) in receiver.lexicon[meaning].items() if value > 0} for meaning in receiver.lexicon}

        # Remove homonyms and synonyms from the lexicon when there are more than 2 entries
        remove_homonyms_and_synonyms(self.lexicon)
        remove_homonyms_and_synonyms(receiver.lexicon)

        # Count the amount of compositional and holistic MUMs in the lexicon
        self.compositional_rules = sum([1 for i in self.lexicon.values() for j in i.values() if j[1] == 'COMPOSITIONAL'])
        self.holistic_rules = sum([1 for i in self.lexicon.values() for j in i.values() if j[1] == 'HOLISTIC'])
        receiver.compositional_rules = sum([1 for i in receiver.lexicon.values() for j in i.values() if j[1] == 'COMPOSITIONAL'])
        receiver.holistic_rules = sum([1 for i in receiver.lexicon.values() for j in i.values() if j[1] == 'HOLISTIC'])

### PATTERN FRAGMENTER ###

class PatternFragmenter:

    def __init__(self, initial_lexicon):
        self.type = 'Pattern Fragmenter'
        self.communicated_meanings = set()
        self.lexicon = copy.deepcopy(initial_lexicon)
        self.buffer = {}
        self.buffer_size = buffer_size 
        self.phrase_rules = {'VSO': 1, 'VOS': 1, 'SVO': 1, 'SOV': 1, 'OVS': 1, 'OSV': 1}
        self.holistic_rules = 0
        self.compositional_rules = 0
        self.successful_interactions = 0
        self.fitness = 1
        self.age = 0

    ### HELPER FUNCTIONS ###
    def print_buffer(self):
        # Print the meaning with corresponding utterance
        for meaning in self.buffer:
            print(f"{meaning}: {self.buffer[meaning]}", file=sys.stdout, flush=True)

    def make_sentence(self, verb, subj, obj, phrase_rule):
        if phrase_rule == 'VSO':
            return str(verb + subj + obj)
        elif phrase_rule == 'VOS':
            return str(verb + obj + subj)
        elif phrase_rule == 'SVO':
            return str(subj + verb + obj)
        elif phrase_rule == 'SOV':
            return str(subj + obj + verb)
        elif phrase_rule == 'OVS':
            return str(obj + verb + subj)
        elif phrase_rule == 'OSV':
            return str(obj + subj + verb)

    ### PATTERN FRAGMENTATION ALGORITHM ###

    def find_shared_constituents(self):
        shared_constituents = dict()
        meanings = list(self.buffer.keys())
        for meaning1 in meanings:
            for meaning2 in meanings:
                if meaning1 != meaning2:
                    verb1, subj1, obj1 = split_integrated_meaning(meaning1)
                    verb2, subj2, obj2 = split_integrated_meaning(meaning2)
                    shared_constituents[meaning1, meaning2] = [constituent for constituent in [verb1, subj1, obj1] if constituent in [verb2, subj2, obj2]]
        return shared_constituents

    def find_common_patterns(self):
        pattern_freq = Counter()
        unique_patterns = []

        # Loop through each utterance in the buffer
        for meaning in self.buffer:
            for utterance in self.buffer[meaning]:
                # Loop through each possible pattern in the string
                for utterance, frequency in self.buffer[meaning].items():
                    for i in range(len(utterance)):
                        search_window = random.randint(3, 5)
                        for j in range(i + search_window, len(utterance) + 1):
                            pattern = utterance[i:j]
                            pattern_freq[pattern] += 1

                            # Add the pattern to the unique patterns list if it is not already there
                            if pattern not in unique_patterns:
                                unique_patterns.append(pattern)

        # Create a list to store the common patterns
        common_patterns = []

        # Loop through the unique patterns
        for pattern in unique_patterns:
            if pattern_freq[pattern] > 1:
                common_patterns.append(pattern)

        # Remove shorter patterns that have a longer counterpart
        common_patterns = [pattern for pattern in common_patterns if all(pattern not in p or pattern == p for p in common_patterns)]

        return common_patterns

    def pattern_recognition(self):
        shared_constituents = self.find_shared_constituents()
        rules_dict = dict()

        for (meaning1, meaning2), shared_constituent in shared_constituents.items():
            if meaning1 != meaning2 and shared_constituent:
                common_patterns = self.find_common_patterns()
                if common_patterns:
                    for pattern in common_patterns:
                        rules_dict[(shared_constituent[0], pattern)] = 1

        return rules_dict

    ### LEARNING ###  #Note: this is actually the learning for a holistic memorizer
    def learn(self):

        # 1 # Check if the buffer is full
        if len(self.buffer) >= self.buffer_size:

            new_rules = self.pattern_recognition()
            # Check if new_rules is empty
            if new_rules:
                # Get the most frequent rule from new_rules
                if new_rules:
                    max_value = max(new_rules.values())
                    most_frequent_rule = random.choice([key for key, value in new_rules.items() if value == max_value])
                    # Get the constituents of the most frequent rule
                    constituent, pattern = most_frequent_rule
                    # Check if the constituent is already in the lexicon
                    if constituent in self.lexicon:
                        # Add the new rule as a compositional rule to the existing constituent in the lexicon, increase the association strenght
                        if pattern in self.lexicon[constituent]:
                            self.lexicon[constituent][pattern] = (self.lexicon[constituent][pattern][0] + 1, 'COMPOSITIONAL')
                        else:
                            # Create a new entry for the constituent in the lexicon and add the new rule
                            self.lexicon[constituent][pattern] = (base_association_strength, 'COMPOSITIONAL')
                    else:
                        # Create a new entry for the constituent in the lexicon and add the new rule
                        self.lexicon[constituent] = {pattern: (base_association_strength, 'COMPOSITIONAL')}

            # Clear the buffer so the agent can learn from new experiences
            self.buffer = {}
    
    ### COMMUNICATION ###

    def send(self, receiver, semantic_space):
        
        # Select a meaning from the semantic space 
        selected_meaning = str(np.random.choice(list(semantic_space.content)))
        selected_verb, selected_subj, selected_obj = split_integrated_meaning(selected_meaning)
        # Add the selected meaning to the set of communicated meanings
        self.communicated_meanings.add(selected_meaning)

        # OPTION 1: All constituent meanings are in the lexicon of the sender
        # Check if all constituents are in the lexicon of the sender
        if selected_verb in self.lexicon and selected_subj in self.lexicon and selected_obj in self.lexicon:
        # 1A: All constituents are coupled to an utterance
            # Check if all constituent MUMs are in the lexicon
            if self.lexicon[selected_verb] and self.lexicon[selected_subj] and self.lexicon[selected_obj]:
                # Get the constituents with the max value for utterance
                retrieved_verb = [max(self.lexicon[selected_verb], key=self.lexicon[selected_verb].get)]
                retrieved_subj = [max(self.lexicon[selected_subj], key=self.lexicon[selected_subj].get)]
                retrieved_obj = [max(self.lexicon[selected_obj], key=self.lexicon[selected_obj].get)]
                # Create a sentence based on the strongest phrase rule
                phrase_rule_strengths = {rule: strength for rule, strength in self.phrase_rules.items() if strength == max(self.phrase_rules.values())}
                phrase_rule_sender = np.random.choice(list(phrase_rule_strengths.keys()))
                sentence = self.make_sentence(retrieved_verb[0], retrieved_subj[0], retrieved_obj[0], phrase_rule_sender)

                if receiver.type == 'Pattern Fragmenter':
                    # Check if the constituent MUMs are in the lexicon of the receiver
                    if selected_verb in receiver.lexicon and selected_subj in receiver.lexicon and selected_obj in receiver.lexicon:
                        if retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 0.5
                            self.successful_interactions += 0.5
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a verb
                            # Check for transparency
                            if np.random.rand() < transparency:
                                # Meanings can be inferred from the context
                                # Add the MUMs to the lexicon of the receiver
                                receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                # Strengthen the MUMs in the lexicon of the sender
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                receiver.successful_interactions += ((2/3) * 0.5)
                                self.successful_interactions += ((2/3) * 0.5)
                                # Check the strongest phrase rule for the receiver
                                phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                # Check if both agent phrase rules are the same
                                if phrase_rule_receiver == phrase_rule_sender:
                                    # Strengthen the phrase rule in the lexicon of both agents
                                    receiver.phrase_rules[phrase_rule_receiver] += 1
                                    self.phrase_rules[phrase_rule_sender] += 1
                                    # Strengthen the communication success for both agents
                                    receiver.successful_interactions += 0.5
                                    self.successful_interactions += 0.5
                                else:
                                    # Decrease the association strength of both phrase rules for both agents
                                    receiver.phrase_rules[phrase_rule_receiver] -= 1
                                    self.phrase_rules[phrase_rule_sender] -= 1
                            else: # The meaning is not transparent and subj and obj cannot be inferred
                                # Check for cues in the environment
                                if np.random.rand() < cue_availability:
                                    if np.random.rand() < cue_availability:
                                        # Add the MUMs to the lexicon of the receiver
                                        receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        # Strengthen the MUMs in the lexicon of the sender
                                        self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                        self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                        # Update communication success for both agents
                                        receiver.successful_interactions += ((2/3) * 0.5)
                                        self.successful_interactions += ((2/3) * 0.5)
                                        # Check the strongest phrase rule for the receiver
                                        phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                        phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                        # Check if both agent phrase rules are the same
                                        if phrase_rule_receiver == phrase_rule_sender:
                                            # Strengthen the phrase rule in the lexicon of both agents
                                            receiver.phrase_rules[phrase_rule_receiver] += 1
                                            self.phrase_rules[phrase_rule_sender] += 1
                                            # Strengthen the communication success for both agents
                                            receiver.successful_interactions += 0.5
                                            self.successful_interactions += 0.5
                                        else:
                                            # Decrease the association strength of both phrase rules for both agents
                                            receiver.phrase_rules[phrase_rule_receiver] -= 1
                                            self.phrase_rules[phrase_rule_sender] -= 1
                                else: # There is no reliable cue
                                    # Decrease the association strength of the MUMs in the lexicon of the sender
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a subject
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is an object
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Decrease the association strength of the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                elif receiver.type == 'Holistic Memorizer':
                    # Check if the MUM is in the lexicon of the receiver
                    if selected_meaning in receiver.lexicon:
                        if sentence in receiver.lexicon[selected_meaning]:
                            # Strengthen the association strength of the MUM in the lexicon of the receiver
                            receiver.lexicon[selected_meaning][sentence] = (receiver.lexicon[selected_meaning][sentence][0] + 1, 'HOLISTIC')
                            # Strengthen the association strengths of the MUMs for the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 1
                            self.successful_interactions += 1
                        else: # The meaning is in the lexicon, but with a different utterance
                            # Check for cues in the environment
                            if np.random.rand() < cue_availability:
                                if np.random.rand() < cue_reliability:
                                    # The meaning can be inferred from the context
                                    # Add the meaning to the lexicon of the receiver
                                    receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                    # Strengthen the association strengths of the MUMs for the sender
                                    self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                    # Update communication success for both agents
                                    self.successful_interactions += 1
                                    receiver.successful_interactions += 1
                            else: # There is no reliable cue available
                                # Add the MUM to the buffer of the receiver
                                if selected_meaning not in receiver.buffer:
                                    receiver.buffer[selected_meaning] = {sentence: 1}
                                else:
                                    receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)
                    else: # The meaning is not in the lexicon of the receiver
                        # Check for cues
                        if np.random.rand() < cue_availability:
                            if np.random.rand() < cue_reliability:
                                # The meaning can be inferred from the context
                                # Add the meaning to the lexicon of the receiver
                                receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                # Strengthen the association strengths of the MUMs for the sender
                                self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                self.successful_interactions += 1
                                receiver.successful_interactions += 1
                        else: # There is no reliable cue available
                            # Add the MUM to the buffer of the receiver
                            if selected_meaning not in receiver.buffer:
                                receiver.buffer[selected_meaning] = {sentence: 1}
                            else:
                                receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)

        # OPTION 2: The sender has a MUM for the verb and the subject
        elif selected_verb in self.lexicon and selected_subj in self.lexicon and not selected_obj in self.lexicon:
            # Check if all constituent MUMs are in the lexicon
            if self.lexicon[selected_verb] and self.lexicon[selected_subj]:
                # Get the constituents with the max value for utterance
                retrieved_verb = [max(self.lexicon[selected_verb], key=self.lexicon[selected_verb].get)]
                retrieved_subj = [max(self.lexicon[selected_subj], key=self.lexicon[selected_subj].get)]
                # Create a new compositional utterance for the object
                retrieved_obj = [generate_compositional_utterance(syllables, None)]
                # Add the new MUM to the lexicon of the sender
                self.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                # Create a sentence based on the strongest phrase rule
                phrase_rule_strengths = {rule: strength for rule, strength in self.phrase_rules.items() if strength == max(self.phrase_rules.values())}
                phrase_rule_sender = np.random.choice(list(phrase_rule_strengths.keys()))                            
                # Check if both agent phrase rules are the same
                sentence = self.make_sentence(retrieved_verb[0], retrieved_subj[0], retrieved_obj[0], phrase_rule_sender)

                if receiver.type == 'Pattern Fragmenter':
                    # Check if the constituent MUMs are in the lexicon of the receiver
                    if selected_verb in receiver.lexicon and selected_subj in receiver.lexicon and selected_obj in receiver.lexicon:
                        if retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 0.5
                            self.successful_interactions += 0.5
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a verb
                            # Check for transparency
                            if np.random.rand() < transparency:
                                # Meanings can be inferred from the context
                                # Add the MUMs to the lexicon of the receiver
                                receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                # Strengthen the MUMs in the lexicon of the sender
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                receiver.successful_interactions += ((2/3) * 0.5)
                                self.successful_interactions += ((2/3) * 0.5)
                                # Check the strongest phrase rule for the receiver
                                phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                # Check if both agent phrase rules are the same
                                if phrase_rule_receiver == phrase_rule_sender:
                                    # Strengthen the phrase rule in the lexicon of both agents
                                    receiver.phrase_rules[phrase_rule_receiver] += 1
                                    self.phrase_rules[phrase_rule_sender] += 1
                                    # Strengthen the communication success for both agents
                                    receiver.successful_interactions += 0.5
                                    self.successful_interactions += 0.5
                                else:
                                    # Decrease the association strength of both phrase rules for both agents
                                    receiver.phrase_rules[phrase_rule_receiver] -= 1
                                    self.phrase_rules[phrase_rule_sender] -= 1
                            else: # The meaning is not transparent and subj and obj cannot be inferred
                                # Check for cues in the environment
                                if np.random.rand() < cue_availability:
                                    if np.random.rand() < cue_availability:
                                        # Add the MUMs to the lexicon of the receiver
                                        receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        # Strengthen the MUMs in the lexicon of the sender
                                        self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                        self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                        # Update communication success for both agents
                                        receiver.successful_interactions += ((2/3) * 0.5)
                                        self.successful_interactions += ((2/3) * 0.5)
                                        # Check the strongest phrase rule for the receiver
                                        phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                        phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                        # Check if both agent phrase rules are the same
                                        if phrase_rule_receiver == phrase_rule_sender:
                                            # Strengthen the phrase rule in the lexicon of both agents
                                            receiver.phrase_rules[phrase_rule_receiver] += 1
                                            self.phrase_rules[phrase_rule_sender] += 1
                                            # Strengthen the communication success for both agents
                                            receiver.successful_interactions += 0.5
                                            self.successful_interactions += 0.5
                                        else:
                                            # Decrease the association strength of both phrase rules for both agents
                                            receiver.phrase_rules[phrase_rule_receiver] -= 1
                                            self.phrase_rules[phrase_rule_sender] -= 1
                                else: # There is no reliable cue
                                    # Decrease the association strength of the MUMs in the lexicon of the sender
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a subject
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is an object
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Decrease the association strength of the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                elif receiver.type == 'Holistic Memorizer':
                    # Check if the MUM is in the lexicon of the receiver
                    if selected_meaning in receiver.lexicon:
                        if sentence in receiver.lexicon[selected_meaning]:
                            # Strengthen the association strength of the MUM in the lexicon of the receiver
                            receiver.lexicon[selected_meaning][sentence] = (receiver.lexicon[selected_meaning][sentence][0] + 1, 'HOLISTIC')
                            # Strengthen the association strengths of the MUMs for the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 1
                            self.successful_interactions += 1
                        else: # The meaning is in the lexicon, but with a different utterance
                            # Check for cues in the environment
                            if np.random.rand() < cue_availability:
                                if np.random.rand() < cue_reliability:
                                    # The meaning can be inferred from the context
                                    # Add the meaning to the lexicon of the receiver
                                    receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                    # Strengthen the association strengths of the MUMs for the sender
                                    self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                    # Update communication success for both agents
                                    self.successful_interactions += 1
                                    receiver.successful_interactions += 1
                            else: # There is no reliable cue available
                                # Add the MUM to the buffer of the receiver
                                if selected_meaning not in receiver.buffer:
                                    receiver.buffer[selected_meaning] = {sentence: 1}
                                else:
                                    receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)
                    else: # The meaning is not in the lexicon of the receiver
                        # Check for cues
                        if np.random.rand() < cue_availability:
                            if np.random.rand() < cue_reliability:
                                # The meaning can be inferred from the context
                                # Add the meaning to the lexicon of the receiver
                                receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                # Strengthen the association strengths of the MUMs for the sender
                                self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                self.successful_interactions += 1
                                receiver.successful_interactions += 1
                        else: # There is no reliable cue available
                            # Add the MUM to the buffer of the receiver
                            if selected_meaning not in receiver.buffer:
                                receiver.buffer[selected_meaning] = {sentence: 1}
                            else:
                                receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)
   
        # OPTION 3: The sender has a MUM for the verb and the object
        elif selected_verb in self.lexicon and not selected_subj in self.lexicon and selected_obj in self.lexicon:
            # Check if all constituent MUMs are in the lexicon
            if self.lexicon[selected_verb] and self.lexicon[selected_obj]:
                # Get the constituents with the max value for utterance
                retrieved_verb = [max(self.lexicon[selected_verb], key=self.lexicon[selected_verb].get)]
                retrieved_obj = [max(self.lexicon[selected_obj], key=self.lexicon[selected_obj].get)]
                # Create a new compositional utterance for the object
                retrieved_subj = [generate_compositional_utterance(syllables, None)]
                # Add the new MUM to the lexicon of the sender
                self.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                # Create a sentence based on the strongest phrase rule
                phrase_rule_strengths = {rule: strength for rule, strength in self.phrase_rules.items() if strength == max(self.phrase_rules.values())}
                phrase_rule_sender = np.random.choice(list(phrase_rule_strengths.keys()))                            
                # Check if both agent phrase rules are the same
                sentence = self.make_sentence(retrieved_verb[0], retrieved_subj[0], retrieved_obj[0], phrase_rule_sender)

                if receiver.type == 'Pattern Fragmenter':
                    # Check if the constituent MUMs are in the lexicon of the receiver
                    if selected_verb in receiver.lexicon and selected_subj in receiver.lexicon and selected_obj in receiver.lexicon:
                        if retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 0.5
                            self.successful_interactions += 0.5
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a verb
                            # Check for transparency
                            if np.random.rand() < transparency:
                                # Meanings can be inferred from the context
                                # Add the MUMs to the lexicon of the receiver
                                receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                # Strengthen the MUMs in the lexicon of the sender
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                receiver.successful_interactions += ((2/3) * 0.5)
                                self.successful_interactions += ((2/3) * 0.5)
                                # Check the strongest phrase rule for the receiver
                                phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                # Check if both agent phrase rules are the same
                                if phrase_rule_receiver == phrase_rule_sender:
                                    # Strengthen the phrase rule in the lexicon of both agents
                                    receiver.phrase_rules[phrase_rule_receiver] += 1
                                    self.phrase_rules[phrase_rule_sender] += 1
                                    # Strengthen the communication success for both agents
                                    receiver.successful_interactions += 0.5
                                    self.successful_interactions += 0.5
                                else:
                                    # Decrease the association strength of both phrase rules for both agents
                                    receiver.phrase_rules[phrase_rule_receiver] -= 1
                                    self.phrase_rules[phrase_rule_sender] -= 1
                            else: # The meaning is not transparent and subj and obj cannot be inferred
                                # Check for cues in the environment
                                if np.random.rand() < cue_availability:
                                    if np.random.rand() < cue_availability:
                                        # Add the MUMs to the lexicon of the receiver
                                        receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        # Strengthen the MUMs in the lexicon of the sender
                                        self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                        self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                        # Update communication success for both agents
                                        receiver.successful_interactions += ((2/3) * 0.5)
                                        self.successful_interactions += ((2/3) * 0.5)
                                        # Check the strongest phrase rule for the receiver
                                        phrase_rule_receiver = np.random.choice([key for key, value in receiver.phrase_rules.items() if value == max(receiver.phrase_rules.values())])
                                        # Check if both agent phrase rules are the same
                                        if phrase_rule_receiver == phrase_rule_sender:
                                            # Strengthen the phrase rule in the lexicon of both agents
                                            receiver.phrase_rules[phrase_rule_receiver] += 1
                                            self.phrase_rules[phrase_rule_sender] += 1
                                            # Strengthen the communication success for both agents
                                            receiver.successful_interactions += 0.5
                                            self.successful_interactions += 0.5
                                        else:
                                            # Decrease the association strength of both phrase rules for both agents
                                            receiver.phrase_rules[phrase_rule_receiver] -= 1
                                            self.phrase_rules[phrase_rule_sender] -= 1
                                else: # There is no reliable cue
                                    # Decrease the association strength of the MUMs in the lexicon of the sender
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a subject
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is an object
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Decrease the association strength of the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                elif receiver.type == 'Holistic Memorizer':
                    # Check if the MUM is in the lexicon of the receiver
                    if selected_meaning in receiver.lexicon:
                        if sentence in receiver.lexicon[selected_meaning]:
                            # Strengthen the association strength of the MUM in the lexicon of the receiver
                            receiver.lexicon[selected_meaning][sentence] = (receiver.lexicon[selected_meaning][sentence][0] + 1, 'HOLISTIC')
                            # Strengthen the association strengths of the MUMs for the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 1
                            self.successful_interactions += 1
                        else: # The meaning is in the lexicon, but with a different utterance
                            # Check for cues in the environment
                            if np.random.rand() < cue_availability:
                                if np.random.rand() < cue_reliability:
                                    # The meaning can be inferred from the context
                                    # Add the meaning to the lexicon of the receiver
                                    receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                    # Strengthen the association strengths of the MUMs for the sender
                                    self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                    # Update communication success for both agents
                                    self.successful_interactions += 1
                                    receiver.successful_interactions += 1
                            else: # There is no reliable cue available
                                # Add the MUM to the buffer of the receiver
                                if selected_meaning not in receiver.buffer:
                                    receiver.buffer[selected_meaning] = {sentence: 1}
                                else:
                                    receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)
                    else: # The meaning is not in the lexicon of the receiver
                        # Check for cues
                        if np.random.rand() < cue_availability:
                            if np.random.rand() < cue_reliability:
                                # The meaning can be inferred from the context
                                # Add the meaning to the lexicon of the receiver
                                receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                # Strengthen the association strengths of the MUMs for the sender
                                self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                self.successful_interactions += 1
                                receiver.successful_interactions += 1
                        else: # There is no reliable cue available
                            # Add the MUM to the buffer of the receiver
                            if selected_meaning not in receiver.buffer:
                                receiver.buffer[selected_meaning] = {sentence: 1}
                            else:
                                receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)

        # OPTION 4: The sender has a MUM for the subj and the object
        elif not selected_verb in self.lexicon and  selected_subj in self.lexicon and selected_obj in self.lexicon:
            # Check if all constituent MUMs are in the lexicon
            if self.lexicon[selected_subj] and self.lexicon[selected_obj]:
                # Get the constituents with the max value for utterance
                retrieved_subj = [max(self.lexicon[selected_subj], key=self.lexicon[selected_subj].get)]
                retrieved_obj = [max(self.lexicon[selected_obj], key=self.lexicon[selected_obj].get)]
                # Create a new compositional utterance for the object
                retrieved_verb = [generate_compositional_utterance(syllables, None)]
                # Add the new MUM to the lexicon of the sender
                self.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                # Create a sentence based on the strongest phrase rule
                phrase_rule_strengths = {rule: strength for rule, strength in self.phrase_rules.items() if strength == max(self.phrase_rules.values())}
                phrase_rule_sender = np.random.choice(list(phrase_rule_strengths.keys()))                            
                # Check if both agent phrase rules are the same
                sentence = self.make_sentence(retrieved_verb[0], retrieved_subj[0], retrieved_obj[0], phrase_rule_sender)

                if receiver.type == 'Pattern Fragmenter':
                    # Check if the constituent MUMs are in the lexicon of the receiver
                    if selected_verb in receiver.lexicon and selected_subj in receiver.lexicon and selected_obj in receiver.lexicon:
                        if retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 0.5
                            self.successful_interactions += 0.5
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a verb
                            # Check for transparency
                            if np.random.rand() < transparency:
                                # Meanings can be inferred from the context
                                # Add the MUMs to the lexicon of the receiver
                                receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                # Strengthen the MUMs in the lexicon of the sender
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                receiver.successful_interactions += ((2/3) * 0.5)
                                self.successful_interactions += ((2/3) * 0.5)
                                # Check the strongest phrase rule for the receiver
                                phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                # Check if both agent phrase rules are the same
                                if phrase_rule_receiver == phrase_rule_sender:
                                    # Strengthen the phrase rule in the lexicon of both agents
                                    receiver.phrase_rules[phrase_rule_receiver] += 1
                                    self.phrase_rules[phrase_rule_sender] += 1
                                    # Strengthen the communication success for both agents
                                    receiver.successful_interactions += 0.5
                                    self.successful_interactions += 0.5
                                else:
                                    # Decrease the association strength of both phrase rules for both agents
                                    receiver.phrase_rules[phrase_rule_receiver] -= 1
                                    self.phrase_rules[phrase_rule_sender] -= 1
                            else: # The meaning is not transparent and subj and obj cannot be inferred
                                # Check for cues in the environment
                                if np.random.rand() < cue_availability:
                                    if np.random.rand() < cue_availability:
                                        # Add the MUMs to the lexicon of the receiver
                                        receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        # Strengthen the MUMs in the lexicon of the sender
                                        self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                        self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                        # Update communication success for both agents
                                        receiver.successful_interactions += ((2/3) * 0.5)
                                        self.successful_interactions += ((2/3) * 0.5)
                                        # Check the strongest phrase rule for the receiver
                                        phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                        phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                        # Check if both agent phrase rules are the same
                                        if phrase_rule_receiver == phrase_rule_sender:
                                            # Strengthen the phrase rule in the lexicon of both agents
                                            receiver.phrase_rules[phrase_rule_receiver] += 1
                                            self.phrase_rules[phrase_rule_sender] += 1
                                            # Strengthen the communication success for both agents
                                            receiver.successful_interactions += 0.5
                                            self.successful_interactions += 0.5
                                        else:
                                            # Decrease the association strength of both phrase rules for both agents
                                            receiver.phrase_rules[phrase_rule_receiver] -= 1
                                            self.phrase_rules[phrase_rule_sender] -= 1
                                else: # There is no reliable cue
                                    # Decrease the association strength of the MUMs in the lexicon of the sender
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a subject
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is an object
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Decrease the association strength of the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                elif receiver.type == 'Holistic Memorizer':
                    # Check if the MUM is in the lexicon of the receiver
                    if selected_meaning in receiver.lexicon:
                        if sentence in receiver.lexicon[selected_meaning]:
                            # Strengthen the association strength of the MUM in the lexicon of the receiver
                            receiver.lexicon[selected_meaning][sentence] = (receiver.lexicon[selected_meaning][sentence][0] + 1, 'HOLISTIC')
                            # Strengthen the association strengths of the MUMs for the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 1
                            self.successful_interactions += 1
                        else: # The meaning is in the lexicon, but with a different utterance
                            # Check for cues in the environment
                            if np.random.rand() < cue_availability:
                                if np.random.rand() < cue_reliability:
                                    # The meaning can be inferred from the context
                                    # Add the meaning to the lexicon of the receiver
                                    receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                    # Strengthen the association strengths of the MUMs for the sender
                                    self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                    # Update communication success for both agents
                                    self.successful_interactions += 1
                                    receiver.successful_interactions += 1
                            else: # There is no reliable cue available
                                # Add the MUM to the buffer of the receiver
                                if selected_meaning not in receiver.buffer:
                                    receiver.buffer[selected_meaning] = {sentence: 1}
                                else:
                                    receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)
                    else: # The meaning is not in the lexicon of the receiver
                        # Check for cues
                        if np.random.rand() < cue_availability:
                            if np.random.rand() < cue_reliability:
                                # The meaning can be inferred from the context
                                # Add the meaning to the lexicon of the receiver
                                receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                # Strengthen the association strengths of the MUMs for the sender
                                self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                self.successful_interactions += 1
                                receiver.successful_interactions += 1
                        else: # There is no reliable cue available
                            # Add the MUM to the buffer of the receiver
                            if selected_meaning not in receiver.buffer:
                                receiver.buffer[selected_meaning] = {sentence: 1}
                            else:
                                receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)

        # OPTION 5: The sender has a MUM for the verb
        elif selected_verb in self.lexicon and not selected_subj in self.lexicon and not selected_obj in self.lexicon:
            # Check if all constituent MUMs are in the lexicon
            if self.lexicon[selected_verb]:
                # Get the constituents with the max value for utterance
                retrieved_verb = [max(self.lexicon[selected_verb], key=self.lexicon[selected_verb].get)]
                # Create a new compositional utterance for the object
                retrieved_subj = [generate_compositional_utterance(syllables, None)]
                retrieved_obj = [generate_compositional_utterance(syllables, None)]
                # Add the new MUM to the lexicon of the sender
                self.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                self.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                # Create a sentence based on the strongest phrase rule
                phrase_rule_strengths = {rule: strength for rule, strength in self.phrase_rules.items() if strength == max(self.phrase_rules.values())}
                phrase_rule_sender = np.random.choice(list(phrase_rule_strengths.keys()))                            
                # Check if both agent phrase rules are the same
                sentence = self.make_sentence(retrieved_verb[0], retrieved_subj[0], retrieved_obj[0], phrase_rule_sender)

                if receiver.type == 'Pattern Fragmenter':
                    # Check if the constituent MUMs are in the lexicon of the receiver
                    if selected_verb in receiver.lexicon and selected_subj in receiver.lexicon and selected_obj in receiver.lexicon:
                        if retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 0.5
                            self.successful_interactions += 0.5
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a verb
                            # Check for transparency
                            if np.random.rand() < transparency:
                                # Meanings can be inferred from the context
                                # Add the MUMs to the lexicon of the receiver
                                receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                # Strengthen the MUMs in the lexicon of the sender
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                receiver.successful_interactions += ((2/3) * 0.5)
                                self.successful_interactions += ((2/3) * 0.5)
                                # Check the strongest phrase rule for the receiver
                                phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                # Check if both agent phrase rules are the same
                                if phrase_rule_receiver == phrase_rule_sender:
                                    # Strengthen the phrase rule in the lexicon of both agents
                                    receiver.phrase_rules[phrase_rule_receiver] += 1
                                    self.phrase_rules[phrase_rule_sender] += 1
                                    # Strengthen the communication success for both agents
                                    receiver.successful_interactions += 0.5
                                    self.successful_interactions += 0.5
                                else:
                                    # Decrease the association strength of both phrase rules for both agents
                                    receiver.phrase_rules[phrase_rule_receiver] -= 1
                                    self.phrase_rules[phrase_rule_sender] -= 1
                            else: # The meaning is not transparent and subj and obj cannot be inferred
                                # Check for cues in the environment
                                if np.random.rand() < cue_availability:
                                    if np.random.rand() < cue_availability:
                                        # Add the MUMs to the lexicon of the receiver
                                        receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        # Strengthen the MUMs in the lexicon of the sender
                                        self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                        self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                        # Update communication success for both agents
                                        receiver.successful_interactions += ((2/3) * 0.5)
                                        self.successful_interactions += ((2/3) * 0.5)
                                        # Check the strongest phrase rule for the receiver
                                        phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                        phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                        # Check if both agent phrase rules are the same
                                        if phrase_rule_receiver == phrase_rule_sender:
                                            # Strengthen the phrase rule in the lexicon of both agents
                                            receiver.phrase_rules[phrase_rule_receiver] += 1
                                            self.phrase_rules[phrase_rule_sender] += 1
                                            # Strengthen the communication success for both agents
                                            receiver.successful_interactions += 0.5
                                            self.successful_interactions += 0.5
                                        else:
                                            # Decrease the association strength of both phrase rules for both agents
                                            receiver.phrase_rules[phrase_rule_receiver] -= 1
                                            self.phrase_rules[phrase_rule_sender] -= 1
                                else: # There is no reliable cue
                                    # Decrease the association strength of the MUMs in the lexicon of the sender
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a subject
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is an object
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Decrease the association strength of the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                elif receiver.type == 'Holistic Memorizer':
                    # Check if the MUM is in the lexicon of the receiver
                    if selected_meaning in receiver.lexicon:
                        if sentence in receiver.lexicon[selected_meaning]:
                            # Strengthen the association strength of the MUM in the lexicon of the receiver
                            receiver.lexicon[selected_meaning][sentence] = (receiver.lexicon[selected_meaning][sentence][0] + 1, 'HOLISTIC')
                            # Strengthen the association strengths of the MUMs for the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 1
                            self.successful_interactions += 1
                        else: # The meaning is in the lexicon, but with a different utterance
                            # Check for cues in the environment
                            if np.random.rand() < cue_availability:
                                if np.random.rand() < cue_reliability:
                                    # The meaning can be inferred from the context
                                    # Add the meaning to the lexicon of the receiver
                                    receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                    # Strengthen the association strengths of the MUMs for the sender
                                    self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                    # Update communication success for both agents
                                    self.successful_interactions += 1
                                    receiver.successful_interactions += 1
                            else: # There is no reliable cue available
                                # Add the MUM to the buffer of the receiver
                                if selected_meaning not in receiver.buffer:
                                    receiver.buffer[selected_meaning] = {sentence: 1}
                                else:
                                    receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)
                    else: # The meaning is not in the lexicon of the receiver
                        # Check for cues
                        if np.random.rand() < cue_availability:
                            if np.random.rand() < cue_reliability:
                                # The meaning can be inferred from the context
                                # Add the meaning to the lexicon of the receiver
                                receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                # Strengthen the association strengths of the MUMs for the sender
                                self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                self.successful_interactions += 1
                                receiver.successful_interactions += 1
                        else: # There is no reliable cue available
                            # Add the MUM to the buffer of the receiver
                            if selected_meaning not in receiver.buffer:
                                receiver.buffer[selected_meaning] = {sentence: 1}
                            else:
                                receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)

        # OPTION 6: The sender has a MUM for the subj
        elif not selected_verb in self.lexicon and selected_subj in self.lexicon and not selected_obj in self.lexicon:
            # Check if all constituent MUMs are in the lexicon
            if self.lexicon[selected_subj]:
                # Get the constituents with the max value for utterance
                retrieved_subj = [max(self.lexicon[selected_subj], key=self.lexicon[selected_subj].get)]
                # Create a new compositional utterance for the object
                retrieved_verb = [generate_compositional_utterance(syllables, None)]
                retrieved_obj = [generate_compositional_utterance(syllables, None)]
                # Add the new MUM to the lexicon of the sender
                self.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                self.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                # Create a sentence based on the strongest phrase rule
                phrase_rule_strengths = {rule: strength for rule, strength in self.phrase_rules.items() if strength == max(self.phrase_rules.values())}
                phrase_rule_sender = np.random.choice(list(phrase_rule_strengths.keys()))                            
                # Check if both agent phrase rules are the same
                sentence = self.make_sentence(retrieved_verb[0], retrieved_subj[0], retrieved_obj[0], phrase_rule_sender)

                if receiver.type == 'Pattern Fragmenter':
                    # Check if the constituent MUMs are in the lexicon of the receiver
                    if selected_verb in receiver.lexicon and selected_subj in receiver.lexicon and selected_obj in receiver.lexicon:
                        if retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 0.5
                            self.successful_interactions += 0.5
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a verb
                            # Check for transparency
                            if np.random.rand() < transparency:
                                # Meanings can be inferred from the context
                                # Add the MUMs to the lexicon of the receiver
                                receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                # Strengthen the MUMs in the lexicon of the sender
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                receiver.successful_interactions += ((2/3) * 0.5)
                                self.successful_interactions += ((2/3) * 0.5)
                                # Check the strongest phrase rule for the receiver
                                phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                # Check if both agent phrase rules are the same
                                if phrase_rule_receiver == phrase_rule_sender:
                                    # Strengthen the phrase rule in the lexicon of both agents
                                    receiver.phrase_rules[phrase_rule_receiver] += 1
                                    self.phrase_rules[phrase_rule_sender] += 1
                                    # Strengthen the communication success for both agents
                                    receiver.successful_interactions += 0.5
                                    self.successful_interactions += 0.5
                                else:
                                    # Decrease the association strength of both phrase rules for both agents
                                    receiver.phrase_rules[phrase_rule_receiver] -= 1
                                    self.phrase_rules[phrase_rule_sender] -= 1
                            else: # The meaning is not transparent and subj and obj cannot be inferred
                                # Check for cues in the environment
                                if np.random.rand() < cue_availability:
                                    if np.random.rand() < cue_availability:
                                        # Add the MUMs to the lexicon of the receiver
                                        receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        # Strengthen the MUMs in the lexicon of the sender
                                        self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                        self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                        # Update communication success for both agents
                                        receiver.successful_interactions += ((2/3) * 0.5)
                                        self.successful_interactions += ((2/3) * 0.5)
                                        # Check the strongest phrase rule for the receiver
                                        phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                        phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                        # Check if both agent phrase rules are the same
                                        if phrase_rule_receiver == phrase_rule_sender:
                                            # Strengthen the phrase rule in the lexicon of both agents
                                            receiver.phrase_rules[phrase_rule_receiver] += 1
                                            self.phrase_rules[phrase_rule_sender] += 1
                                            # Strengthen the communication success for both agents
                                            receiver.successful_interactions += 0.5
                                            self.successful_interactions += 0.5
                                        else:
                                            # Decrease the association strength of both phrase rules for both agents
                                            receiver.phrase_rules[phrase_rule_receiver] -= 1
                                            self.phrase_rules[phrase_rule_sender] -= 1
                                else: # There is no reliable cue
                                    # Decrease the association strength of the MUMs in the lexicon of the sender
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a subject
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is an object
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Decrease the association strength of the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                elif receiver.type == 'Holistic Memorizer':
                    # Check if the MUM is in the lexicon of the receiver
                    if selected_meaning in receiver.lexicon:
                        if sentence in receiver.lexicon[selected_meaning]:
                            # Strengthen the association strength of the MUM in the lexicon of the receiver
                            receiver.lexicon[selected_meaning][sentence] = (receiver.lexicon[selected_meaning][sentence][0] + 1, 'HOLISTIC')
                            # Strengthen the association strengths of the MUMs for the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 1
                            self.successful_interactions += 1
                        else: # The meaning is in the lexicon, but with a different utterance
                            # Check for cues in the environment
                            if np.random.rand() < cue_availability:
                                if np.random.rand() < cue_reliability:
                                    # The meaning can be inferred from the context
                                    # Add the meaning to the lexicon of the receiver
                                    receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                    # Strengthen the association strengths of the MUMs for the sender
                                    self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                    # Update communication success for both agents
                                    self.successful_interactions += 1
                                    receiver.successful_interactions += 1
                            else: # There is no reliable cue available
                                # Add the MUM to the buffer of the receiver
                                if selected_meaning not in receiver.buffer:
                                    receiver.buffer[selected_meaning] = {sentence: 1}
                                else:
                                    receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)
                    else: # The meaning is not in the lexicon of the receiver
                        # Check for cues
                        if np.random.rand() < cue_availability:
                            if np.random.rand() < cue_reliability:
                                # The meaning can be inferred from the context
                                # Add the meaning to the lexicon of the receiver
                                receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                # Strengthen the association strengths of the MUMs for the sender
                                self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                self.successful_interactions += 1
                                receiver.successful_interactions += 1
                        else: # There is no reliable cue available
                            # Add the MUM to the buffer of the receiver
                            if selected_meaning not in receiver.buffer:
                                receiver.buffer[selected_meaning] = {sentence: 1}
                            else:
                                receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)

        # OPTION 7: The sender has a MUM for the obj
        elif not selected_verb in self.lexicon and not selected_subj in self.lexicon and selected_obj in self.lexicon:
            # Check if all constituent MUMs are in the lexicon
            if self.lexicon[selected_obj]:
                # Get the constituents with the max value for utterance
                retrieved_obj = [max(self.lexicon[selected_obj], key=self.lexicon[selected_obj].get)]
                # Create a new compositional utterance for the object
                retrieved_verb = [generate_compositional_utterance(syllables, None)]
                retrieved_subj = [generate_compositional_utterance(syllables, None)]
                # Add the new MUM to the lexicon of the sender
                self.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                self.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                # Create a sentence based on the strongest phrase rule
                phrase_rule_strengths = {rule: strength for rule, strength in self.phrase_rules.items() if strength == max(self.phrase_rules.values())}
                phrase_rule_sender = np.random.choice(list(phrase_rule_strengths.keys()))                            
                # Check if both agent phrase rules are the same
                sentence = self.make_sentence(retrieved_verb[0], retrieved_subj[0], retrieved_obj[0], phrase_rule_sender)

                if receiver.type == 'Pattern Fragmenter':
                    # Check if the constituent MUMs are in the lexicon of the receiver
                    if selected_verb in receiver.lexicon and selected_subj in receiver.lexicon and selected_obj in receiver.lexicon:
                        if retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 0.5
                            self.successful_interactions += 0.5
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a verb
                            # Check for transparency
                            if np.random.rand() < transparency:
                                # Meanings can be inferred from the context
                                # Add the MUMs to the lexicon of the receiver
                                receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                # Strengthen the MUMs in the lexicon of the sender
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                receiver.successful_interactions += ((2/3) * 0.5)
                                self.successful_interactions += ((2/3) * 0.5)
                                # Check the strongest phrase rule for the receiver
                                phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                # Check if both agent phrase rules are the same
                                if phrase_rule_receiver == phrase_rule_sender:
                                    # Strengthen the phrase rule in the lexicon of both agents
                                    receiver.phrase_rules[phrase_rule_receiver] += 1
                                    self.phrase_rules[phrase_rule_sender] += 1
                                    # Strengthen the communication success for both agents
                                    receiver.successful_interactions += 0.5
                                    self.successful_interactions += 0.5
                                else:
                                    # Decrease the association strength of both phrase rules for both agents
                                    receiver.phrase_rules[phrase_rule_receiver] -= 1
                                    self.phrase_rules[phrase_rule_sender] -= 1
                            else: # The meaning is not transparent and subj and obj cannot be inferred
                                # Check for cues in the environment
                                if np.random.rand() < cue_availability:
                                    if np.random.rand() < cue_availability:
                                        # Add the MUMs to the lexicon of the receiver
                                        receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        # Strengthen the MUMs in the lexicon of the sender
                                        self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                        self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                        # Update communication success for both agents
                                        receiver.successful_interactions += ((2/3) * 0.5)
                                        self.successful_interactions += ((2/3) * 0.5)
                                        # Check the strongest phrase rule for the receiver
                                        phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                        phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                        # Check if both agent phrase rules are the same
                                        if phrase_rule_receiver == phrase_rule_sender:
                                            # Strengthen the phrase rule in the lexicon of both agents
                                            receiver.phrase_rules[phrase_rule_receiver] += 1
                                            self.phrase_rules[phrase_rule_sender] += 1
                                            # Strengthen the communication success for both agents
                                            receiver.successful_interactions += 0.5
                                            self.successful_interactions += 0.5
                                        else:
                                            # Decrease the association strength of both phrase rules for both agents
                                            receiver.phrase_rules[phrase_rule_receiver] -= 1
                                            self.phrase_rules[phrase_rule_sender] -= 1
                                else: # There is no reliable cue
                                    # Decrease the association strength of the MUMs in the lexicon of the sender
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a subject
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is an object
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Decrease the association strength of the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                elif receiver.type == 'Holistic Memorizer':
                    # Check if the MUM is in the lexicon of the receiver
                    if selected_meaning in receiver.lexicon:
                        if sentence in receiver.lexicon[selected_meaning]:
                            # Strengthen the association strength of the MUM in the lexicon of the receiver
                            receiver.lexicon[selected_meaning][sentence] = (receiver.lexicon[selected_meaning][sentence][0] + 1, 'HOLISTIC')
                            # Strengthen the association strengths of the MUMs for the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 1
                            self.successful_interactions += 1
                        else: # The meaning is in the lexicon, but with a different utterance
                            # Check for cues in the environment
                            if np.random.rand() < cue_availability:
                                if np.random.rand() < cue_reliability:
                                    # The meaning can be inferred from the context
                                    # Add the meaning to the lexicon of the receiver
                                    receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                    # Strengthen the association strengths of the MUMs for the sender
                                    self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                    # Update communication success for both agents
                                    self.successful_interactions += 1
                                    receiver.successful_interactions += 1
                            else: # There is no reliable cue available
                                # Add the MUM to the buffer of the receiver
                                if selected_meaning not in receiver.buffer:
                                    receiver.buffer[selected_meaning] = {sentence: 1}
                                else:
                                    receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)
                    else: # The meaning is not in the lexicon of the receiver
                        # Check for cues
                        if np.random.rand() < cue_availability:
                            if np.random.rand() < cue_reliability:
                                # The meaning can be inferred from the context
                                # Add the meaning to the lexicon of the receiver
                                receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                # Strengthen the association strengths of the MUMs for the sender
                                self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                self.successful_interactions += 1
                                receiver.successful_interactions += 1
                        else: # There is no reliable cue available
                            # Add the MUM to the buffer of the receiver
                            if selected_meaning not in receiver.buffer:
                                receiver.buffer[selected_meaning] = {sentence: 1}
                            else:
                                receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)

        # OPTION 8: The sender has no compositional MUMs
        elif not selected_verb in self.lexicon and not selected_subj in self.lexicon and not selected_obj in self.lexicon:
            # Check if the sender already has a compositional rule in its lexicon
            if self.compositional_rules == 0:
                # Check if there is a holistic MUM for the selected meaning
                if selected_meaning in self.lexicon:
                    if self.lexicon[selected_meaning]:
                        # Retrieve the MUM with the highest association strength
                        retrieved_utterance = [max(self.lexicon[selected_meaning], key=self.lexicon[selected_meaning].get)]
                    else:
                        retrieved_utterance = [generate_holistic_utterance(syllables, None)]
                        self.lexicon[selected_meaning] = {retrieved_utterance[0]: (base_association_strength, 'HOLISTIC')}
                else:
                    retrieved_utterance = [generate_holistic_utterance(syllables, None)]
                    self.lexicon[selected_meaning] = {retrieved_utterance[0]: (base_association_strength, 'HOLISTIC')}

                # Communicate the MUM to the receiver (communication is the same for both receiver types)

                # Check if the MUM is in the lexicon of the receiver
                if selected_meaning in receiver.lexicon:
                    if retrieved_utterance[0] in receiver.lexicon[selected_meaning]:
                        # Strengthen the association strength of the MUM in the lexicon of both agents
                        receiver.lexicon[selected_meaning][retrieved_utterance[0]] = (receiver.lexicon[selected_meaning][retrieved_utterance[0]][0] + 1, 'HOLISTIC')
                        self.lexicon[selected_meaning][retrieved_utterance[0]] = (self.lexicon[selected_meaning][retrieved_utterance[0]][0] + 1, 'HOLISTIC')
                        # Update communication success for both agents
                        receiver.successful_interactions += 1
                        self.successful_interactions += 1
                    else: # The meaning is in the lexicon, but with a different utterance
                        # Check for cues in the environment
                        if np.random.rand() < cue_availability:
                            if np.random.rand() < cue_reliability:
                                # The meaning can be inferred from the context
                                # Add the meaning to the lexicon of the receiver
                                receiver.lexicon[selected_meaning] = {retrieved_utterance[0]: (base_association_strength, 'HOLISTIC')}
                                # Strengthen the MUM in the lexicon of the sender
                                self.lexicon[selected_meaning][retrieved_utterance[0]] = (self.lexicon[selected_meaning][retrieved_utterance[0]][0] + 1, 'HOLISTIC')
                                # Update communication success for both agents
                                self.successful_interactions += 1
                                receiver.successful_interactions += 1
                        else: # There is no reliable cue available
                            # Add the MUM to the buffer of the receiver
                            if selected_meaning not in receiver.buffer:
                                receiver.buffer[selected_meaning] = {retrieved_utterance[0]: 1}
                            else:
                                receiver.buffer[selected_meaning][retrieved_utterance[0]] = (receiver.buffer[selected_meaning].get(retrieved_utterance[0], 0) + 1)
                else: # The meaning is not in the lexicon of the receiver
                    # Check for cues
                    if np.random.rand() < cue_availability:
                        if np.random.rand() < cue_reliability:
                            # The meaning can be inferred from the context
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_meaning] = {retrieved_utterance[0]: (base_association_strength, 'HOLISTIC')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_meaning][retrieved_utterance[0]] = (self.lexicon[selected_meaning][retrieved_utterance[0]][0] + 1, 'HOLISTIC')
                            # Update communication success for both agents
                            self.successful_interactions += 1
                            receiver.successful_interactions += 1
                    else: # There is no reliable cue available
                        # Add the MUM to the buffer of the receiver
                        if selected_meaning not in receiver.buffer:
                            receiver.buffer[selected_meaning] = {retrieved_utterance[0]: 1}
                        else:
                            receiver.buffer[selected_meaning][retrieved_utterance[0]] = (receiver.buffer[selected_meaning].get(retrieved_utterance[0], 0) + 1)
            
            else: # The agent has already 'discovered' compositionality

                # Create a new compositional utterance for the object
                retrieved_verb = [generate_compositional_utterance(syllables, None)]
                retrieved_subj = [generate_compositional_utterance(syllables, None)]
                retrieved_obj = [generate_compositional_utterance(syllables, None)]
                # Add the new MUM to the lexicon of the sender
                self.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                self.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                self.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                # Create a sentence based on the strongest phrase rule
                phrase_rule_strengths = {rule: strength for rule, strength in self.phrase_rules.items() if strength == max(self.phrase_rules.values())}
                phrase_rule_sender = np.random.choice(list(phrase_rule_strengths.keys()))                            
                # Check if both agent phrase rules are the same
                sentence = self.make_sentence(retrieved_verb[0], retrieved_subj[0], retrieved_obj[0], phrase_rule_sender)

                if receiver.type == 'Pattern Fragmenter':
                    # Check if the constituent MUMs are in the lexicon of the receiver
                    if selected_verb in receiver.lexicon and selected_subj in receiver.lexicon and selected_obj in receiver.lexicon:
                        if retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 0.5
                            self.successful_interactions += 0.5
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1
                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success of both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUMs in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # 2 constituents are present, so the 3rd one can be inferred
                            # Add the meaning to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUM in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_verb][retrieved_verb[0]] = (receiver.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a verb
                            # Check for transparency
                            if np.random.rand() < transparency:
                                # Meanings can be inferred from the context
                                # Add the MUMs to the lexicon of the receiver
                                receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                # Strengthen the MUMs in the lexicon of the sender
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                receiver.successful_interactions += ((2/3) * 0.5)
                                self.successful_interactions += ((2/3) * 0.5)
                                # Check the strongest phrase rule for the receiver
                                phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                # Check if both agent phrase rules are the same
                                if phrase_rule_receiver == phrase_rule_sender:
                                    # Strengthen the phrase rule in the lexicon of both agents
                                    receiver.phrase_rules[phrase_rule_receiver] += 1
                                    self.phrase_rules[phrase_rule_sender] += 1
                                    # Strengthen the communication success for both agents
                                    receiver.successful_interactions += 0.5
                                    self.successful_interactions += 0.5
                                else:
                                    # Decrease the association strength of both phrase rules for both agents
                                    receiver.phrase_rules[phrase_rule_receiver] -= 1
                                    self.phrase_rules[phrase_rule_sender] -= 1
                            else: # The meaning is not transparent and subj and obj cannot be inferred
                                # Check for cues in the environment
                                if np.random.rand() < cue_availability:
                                    if np.random.rand() < cue_availability:
                                        # Add the MUMs to the lexicon of the receiver
                                        receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                                        # Strengthen the MUMs in the lexicon of the sender
                                        self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                                        self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                                        # Update communication success for both agents
                                        receiver.successful_interactions += ((2/3) * 0.5)
                                        self.successful_interactions += ((2/3) * 0.5)
                                        # Check the strongest phrase rule for the receiver
                                        phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                                        phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                                        # Check if both agent phrase rules are the same
                                        if phrase_rule_receiver == phrase_rule_sender:
                                            # Strengthen the phrase rule in the lexicon of both agents
                                            receiver.phrase_rules[phrase_rule_receiver] += 1
                                            self.phrase_rules[phrase_rule_sender] += 1
                                            # Strengthen the communication success for both agents
                                            receiver.successful_interactions += 0.5
                                            self.successful_interactions += 0.5
                                        else:
                                            # Decrease the association strength of both phrase rules for both agents
                                            receiver.phrase_rules[phrase_rule_receiver] -= 1
                                            self.phrase_rules[phrase_rule_sender] -= 1
                                else: # There is no reliable cue
                                    # Decrease the association strength of the MUMs in the lexicon of the sender
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_subj][retrieved_subj[0]] = (receiver.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is a subject
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_obj] = {retrieved_obj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Strengthen the association strength of the MUM in the lexicon of both agents
                            receiver.lexicon[selected_obj][retrieved_obj[0]] = (receiver.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((1/3) * 0.5)
                            self.successful_interactions += ((1/3) * 0.5)
                            # 1 constituent is present, and it is an object
                            # Meanings can be inferred from context
                            # Add the MUMs to the lexicon of the receiver
                            receiver.lexicon[selected_verb] = {retrieved_verb[0]: (base_association_strength, 'COMPOSITIONAL')}
                            receiver.lexicon[selected_subj] = {retrieved_subj[0]: (base_association_strength, 'COMPOSITIONAL')}
                            # Strengthen the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] + 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] + 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += ((2/3) * 0.5)
                            self.successful_interactions += ((2/3) * 0.5)
                            # Check the strongest phrase rule for the receiver
                            phrase_rule_strengths = {rule: strength for rule, strength in receiver.phrase_rules.items() if strength == max(receiver.phrase_rules.values())}
                            phrase_rule_receiver = np.random.choice(list(phrase_rule_strengths.keys()))                            # Check if both agent phrase rules are the same
                            # Check if both agent phrase rules are the same
                            if phrase_rule_receiver == phrase_rule_sender:
                                # Strengthen the phrase rule in the lexicon of both agents
                                receiver.phrase_rules[phrase_rule_receiver] += 1
                                self.phrase_rules[phrase_rule_sender] += 1
                                # Strengthen the communication success for both agents
                                receiver.successful_interactions += 0.5
                                self.successful_interactions += 0.5
                            else:
                                # Decrease the association strength of both phrase rules for both agents
                                receiver.phrase_rules[phrase_rule_receiver] -= 1
                                self.phrase_rules[phrase_rule_sender] -= 1

                        elif not retrieved_verb[0] in receiver.lexicon[selected_verb] and not retrieved_subj[0] in receiver.lexicon[selected_subj] and not retrieved_obj[0] in receiver.lexicon[selected_obj]:
                            # Decrease the association strength of the MUMs in the lexicon of the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')

                elif receiver.type == 'Holistic Memorizer':
                    # Check if the MUM is in the lexicon of the receiver
                    if selected_meaning in receiver.lexicon:
                        if sentence in receiver.lexicon[selected_meaning]:
                            # Strengthen the association strength of the MUM in the lexicon of the receiver
                            receiver.lexicon[selected_meaning][sentence] = (receiver.lexicon[selected_meaning][sentence][0] + 1, 'HOLISTIC')
                            # Strengthen the association strengths of the MUMs for the sender
                            self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                            self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                            # Update communication success for both agents
                            receiver.successful_interactions += 1
                            self.successful_interactions += 1
                        else: # The meaning is in the lexicon, but with a different utterance
                            # Check for cues in the environment
                            if np.random.rand() < cue_availability:
                                if np.random.rand() < cue_reliability:
                                    # The meaning can be inferred from the context
                                    # Add the meaning to the lexicon of the receiver
                                    receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                    # Strengthen the association strengths of the MUMs for the sender
                                    self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                    self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                    # Update communication success for both agents
                                    self.successful_interactions += 1
                                    receiver.successful_interactions += 1
                            else: # There is no reliable cue available
                                # Add the MUM to the buffer of the receiver
                                if selected_meaning not in receiver.buffer:
                                    receiver.buffer[selected_meaning] = {sentence: 1}
                                else:
                                    receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)
                    else: # The meaning is not in the lexicon of the receiver
                        # Check for cues
                        if np.random.rand() < cue_availability:
                            if np.random.rand() < cue_reliability:
                                # The meaning can be inferred from the context
                                # Add the meaning to the lexicon of the receiver
                                receiver.lexicon[selected_meaning] = {sentence: (base_association_strength, 'HOLISTIC')}
                                # Strengthen the association strengths of the MUMs for the sender
                                self.lexicon[selected_verb][retrieved_verb[0]] = (self.lexicon[selected_verb][retrieved_verb[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_subj][retrieved_subj[0]] = (self.lexicon[selected_subj][retrieved_subj[0]][0] - 1, 'COMPOSITIONAL')
                                self.lexicon[selected_obj][retrieved_obj[0]] = (self.lexicon[selected_obj][retrieved_obj[0]][0] - 1, 'COMPOSITIONAL')
                                # Update communication success for both agents
                                self.successful_interactions += 1
                                receiver.successful_interactions += 1
                        else: # There is no reliable cue available
                            # Add the MUM to the buffer of the receiver
                            if selected_meaning not in receiver.buffer:
                                receiver.buffer[selected_meaning] = {sentence: 1}
                            else:
                                receiver.buffer[selected_meaning][sentence] = (receiver.buffer[selected_meaning].get(sentence, 0) + 1)

        # Learning and updating
        self.learn()
        receiver.learn()

        # Remove MUMs with association strength 0 from the lexicon
        self.lexicon = {meaning: {utterance: (value, m_type) for utterance, (value, m_type) in self.lexicon[meaning].items() if value > 0} for meaning in self.lexicon}
        receiver.lexicon = {meaning: {utterance: (value, m_type) for utterance, (value, m_type) in receiver.lexicon[meaning].items() if value > 0} for meaning in receiver.lexicon}

        # Set phrase rules with value < 1 to 1
        self.phrase_rules = {key: max(1, value) if value < 1 else value for key, value in self.phrase_rules.items()}

        # Remove homonyms and synonyms from the lexicon when there are more than 2 entries
        remove_homonyms_and_synonyms(self.lexicon)
        remove_homonyms_and_synonyms(receiver.lexicon)

        # Count the amount of compositional and holistic MUMs in the lexicon
        self.compositional_rules = sum([1 for i in self.lexicon.values() for j in i.values() if j[1] == 'COMPOSITIONAL'])
        self.holistic_rules = sum([1 for i in self.lexicon.values() for j in i.values() if j[1] == 'HOLISTIC'])
        receiver.compositional_rules = sum([1 for i in receiver.lexicon.values() for j in i.values() if j[1] == 'COMPOSITIONAL'])
        receiver.holistic_rules = sum([1 for i in receiver.lexicon.values() for j in i.values() if j[1] == 'HOLISTIC'])


#=============#
# Naming Game #
#=============#

def naming_game(population, semantic_space):

    for _ in range(communication_rounds):

        # Select a random communicator and receiver
        np.random.shuffle(population)
        sender = np.random.choice(population)
        receiver = np.random.choice(population)

        while sender == receiver:
            receiver = np.random.choice(population)

        # Communicate
        sender.send(receiver, semantic_space)
        receiver.send(sender, semantic_space)


#========================#
# Evolutionary functions #
#========================#

def mutate(agent):
    if np.random.rand() < mutation_rate:
        # Mutate the agent type
        if agent.type == 'Pattern Fragmenter':
            agent = HolisticMemorizer
        else:
            agent = PatternFragmenter

def learn_lexicon(offspring, parent, semantic_space, syllables):
    # If the type of offspring and parent is the same
    if offspring.type == parent.type:
        # Learn the lexicon from the parent
        offspring.lexicon = parent.lexicon.copy()
        # Randomly remove items from the offspring lexicon (does not learn all from parent)
        for meaning in list(offspring.lexicon.keys()):
            for utterance in list(offspring.lexicon[meaning].keys()):
                if np.random.rand() < cultural_inheritance_rate:
                    del offspring.lexicon[meaning][utterance]

    # If the type of offspring and parent is different
    else:
        # Initialize a random lexicon for the offspring
        offspring.lexicon = initialize_lexicon(semantic_space)

def learn_phrase_rules(offspring, parent):
    # If both offspring and parent are pattern fragmenters
    if offspring.type == 'Pattern Fragmenter' and parent.type == 'Pattern Fragmenter':
        # Take the strongest phrase rules of the parent and add them to the own phrase rules
        for rule, strength in parent.phrase_rules.items():
            if rule not in offspring.phrase_rules:
                offspring.phrase_rules[rule] = strength
            else:
                offspring.phrase_rules[rule] += strength

        # Only retain the strongest phrase rule and put the strength of others to 0
        strongest_rule = max(offspring.phrase_rules, key = offspring.phrase_rules.get)
        for rule, strength in offspring.phrase_rules.items():
            if rule != strongest_rule:
                offspring.phrase_rules[rule] = 0

def update_fitness(agent):
    if agent.type == 'Holistic Memorizer':
        agent.fitness = agent.successful_interactions - (len(agent.lexicon) * lexical_cost) 
        if agent.fitness <= 0:
            agent.fitness = 1
    else:
        agent.fitness = agent.successful_interactions - cognitive_cost - (len(agent.lexicon) * lexical_cost)
        if agent.fitness <= 0:
            agent.fitness = 1


#============#
# Simulation #
#============#

def simulation(seed):

    np.random.seed(seed)
    random.seed(seed)

    # Initialize the semantic space
    semantic_space = SemanticSpace()
    semantic_space.initialize_semantic_space()

    # Create an initial lexicon that is shared among all agents
    initial_lexicon = initialize_lexicon(semantic_space)

    # Initialize the population of agents
    population = []

    if proportion_holistic_memorizers + proportion_pattern_fragmenters != 1:
        raise ValueError("The proportions of the agent types do not add up to 1!")
    num_holistic_memorizers = int(population_size * proportion_holistic_memorizers)
    num_pattern_fragmenters = int(population_size * proportion_pattern_fragmenters)

    for _ in range(num_holistic_memorizers):
        population.append(HolisticMemorizer(initial_lexicon))
    for _ in range(num_pattern_fragmenters):
        population.append(PatternFragmenter(initial_lexicon))

    # Assign a random age to each agent
    for agent in population:
        agent.age = np.random.randint(0, max_age)

    # Initialize lists for storing data
    holistic_rules_in_population = []
    compositional_rules_in_population = []
    compositionality = []

    # Run the simulation for a number of generations
    for _ in range(num_generations):
        print(f"Generation: {_ + 1}")
        current_generation = _ + 1

        # Create an empty next generation
        next_generation = []

        # Run the naming game for a number of rounds
        naming_game(population, semantic_space)

        # Calculate the fitness of each agent
        for agent in population:
            update_fitness(agent)

        # Select the parents for the next generation based on age
        parents = [agent for agent in population if agent.age >= reproductive_age]

        # Calculate the relative fitness of all parents in the population
        total_fitness = sum([agent.fitness for agent in parents])

        # Reproduce proportional to the fitness of the agents
        for _ in range(len(parents)):
            # Select a possible parent from the population
            relative_fitness = [agent.fitness / total_fitness for agent in parents]
            parent = np.random.choice(parents, p=relative_fitness, replace=True)

            # Determine the number of offspring for the parent (proportional to fitness)
            num_offspring = int(parent.fitness / total_fitness * len(parents))

            # Create the offspring
            for _ in range(num_offspring):
                if parent.type == 'Holistic Memorizer':
                    offspring = HolisticMemorizer(parent.lexicon)
                else:
                    offspring = PatternFragmenter(parent.lexicon)

                # Mutate the offspring
                mutate(offspring)

                # Learn the lexicon and phrase rules from the parent
                if offspring.type == 'Holistic Memorizer':
                    learn_lexicon(offspring, parent, semantic_space, syllables)
                else:
                    learn_phrase_rules(offspring, parent)
                    learn_lexicon(offspring, parent, semantic_space, syllables)

                # Add the offspring to the next generation
                next_generation.append(offspring)

        # Update the population
        for agent in population:
            agent.age += 1
        population.extend(next_generation)
        population = [agent for agent in population if agent.age <= max_age]

        # If the population is too large, remove random agents, proportional to their fitness
        # Death
        while len(population) > max_population_size:
            # Calculate the relative fitness of all agents in the population
            total_fitness = sum([agent.fitness for agent in population])
            relative_fitness = [agent.fitness / total_fitness for agent in population]  # Fix: Use 'population' instead of 'parents'

            # Select an agent to remove from the population
            agent = np.random.choice(population, p=relative_fitness, replace=False)
            population.remove(agent)


        # Calculate the compositionality of the population
        total_compositional_rules = sum([agent.compositional_rules for agent in population])
        total_holistic_rules = sum([agent.holistic_rules for agent in population])
        if total_compositional_rules + total_holistic_rules != 0:
            compositionality.append(total_compositional_rules / (total_compositional_rules + total_holistic_rules) * 100)
        else:
            compositionality.append(0)


        # Grow the semantic space

        if np.random.rand() < cultural_evolution_rate:
            semantic_space.grow_semantic_space()

        # Change the environment
        if np.random.rand() < environmental_change_rate:
            semantic_space.environment_changes()
    
    # Create a list to store final compositionality
    final_compositionality = []

    # Calculate the compositionality of the final population
    total_compositional_rules = sum([agent.compositional_rules for agent in population])
    total_holistic_rules = sum([agent.holistic_rules for agent in population])
    if total_compositional_rules + total_holistic_rules != 0:
        final_compositionality.append(total_compositional_rules / (total_compositional_rules + total_holistic_rules) * 100)
    else:
        final_compositionality.append(0)

    return final_compositionality

    
#======#
# Main #
#======#

for simnr in range(simulations):
    seed = np.random.randint(0, 100000)
    final_compositionality = simulation(seed)