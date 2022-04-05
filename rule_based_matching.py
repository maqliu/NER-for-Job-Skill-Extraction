import spacy
import numpy as np
from spacy.matcher import Matcher
from fuzzywuzzy import fuzz
import texthero as hero

# nlp = spacy.load("en_core_web_sm")

def sentence_matching(input,nlp):
  matcher = Matcher(nlp.vocab)

  pattern1 = [{"LOWER": {"IN": ["ability","experience","proficient", "familiarity","capability","level","responsible","degree","skill"]}},{"DEP":"prep"}]
  pattern2 = [{"LOWER": "ability"},{"DEP":"aux"}]

  matcher.add("proficient", [pattern1])
  matcher.add("ability", [pattern2])

  doc = nlp(input)#doc here 
  matches = matcher(doc)

  sentence_list = []
  for s in doc.sents:
    sentence_list.append(s.end)

  sentence_list= np.array(sentence_list)
  output_sents = []
  match_info = []
  for match_id, start, end in matches:
      string_id = nlp.vocab.strings[match_id]  # Get string representation
      sents_end = sentence_list[int(np.argwhere(sentence_list>end)[0])]
      span = doc[start:sents_end]  # The matched span
      # print(match_id, string_id, start, end, span.text)
      match_info.append((match_id, string_id, start, end))
      output_sents.append(span.text)
  return match_info, output_sents

def sentence_processing(sentence,nlp):
  doc = nlp(sentence)

  root = [token for token in doc if token.head == token][0]
  subject = list(root.rights)[0]
  hard_skill = []
  for descendant in subject.subtree:
      # assert subject is descendant or subject.is_ancestor(descendant)
      if descendant.dep_ in ['pobj','conj']:
        hard_skill.append(descendant.text)
  return hard_skill

def run_hardskill_extraction(input,nlp):
  match_info, output_sents = sentence_matching(input,nlp)
  output_skills = []
  for sent in output_sents:
    output_skills += sentence_processing(sent,nlp)
  return output_skills


