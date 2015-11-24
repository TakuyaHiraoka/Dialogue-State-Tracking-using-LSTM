import json,types

class OntologyReader():
	def __init__(self, ontology_file_name):
		self.ontology = json.load(open(ontology_file_name, 'r'))

		self.tagsets = self.ontology['tagsets']
		for topic in self.tagsets:
			for slot in self.tagsets[topic]:
				value_list = []
				for value in self.tagsets[topic][slot]:
					if type(value) in types.StringTypes:
						value_list.append(value)
					elif type(value) == types.DictType:
						if 'type' in value and value['type'] == 'knowledge':
							if 'source' in value and value['source'] in self.ontology['knowledge']:
								for item in self.ontology['knowledge'][value['source']]:
									if 'slot' in value and value['slot'] in item:
										value_list.append(item[value['slot']])
				value_list = sorted(set(value_list))
				self.tagsets[topic][slot] = value_list

	def get_topics(self):
		return self.tagsets.keys()

	def get_slots(self, topic):
		result = None
		if topic in self.get_topics():
			result = self.tagsets[topic].keys()
		return result

	def get_tagsets(self):
		return self.tagsets