from deeppavlov.contrib.skills.similarity_matching_skill import SimilarityMatchingSkill
from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector


# hello = PatternMatchingSkill(responses=["Hello world!"], patterns=["(hi|hello|good day)"], regex = True)
# sorry = PatternMatchingSkill(responses=["don't be sorry", "Please don't"], patterns=["(sorry|excuse)"], regex = True)
# perhaps = PatternMatchingSkill(responses=["Please be more specific"], patterns=["(.*)perhaps(.*)"], regex = True)


hello = PatternMatchingSkill(responses=['Hi, I am a chatbot'], patterns=['hi', 'hello', 'How are you'], default_confidence=0.3)
faq = SimilarityMatchingSkill(
	data_path ='http://files.deeppavlov.ai/faq/dataset.csv',
	x_col_name = 'Question',
	y_col_name = 'Answer',
	save_load_path = './model',
	config_type = 'tfidf_autofaq',
	edit_dict = {},
	train = False
)
bye = PatternMatchingSkill(responses=['have a nice day'], patterns=['bye', 'goodbye'], default_confidence=0.3)

agent = DefaultAgent([hello, bye, faq], skills_selector=HighestConfidenceSelector())


# agent = DefaultAgent([hello, sorry, perhaps], skills_selector=HighestConfidenceSelector())
# q = agent(['hi, how are you', 'I am sorry', 'perhaps I am not sure'])

q = agent(['Hello', 'How can I get a visa?'])

print(q)
