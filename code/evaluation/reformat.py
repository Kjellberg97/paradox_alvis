



ex = [['{\'proof\': {\'[["loving"], "unpleasant"]\': {\',loving\': 1}}}, \'label\': 1}', 
  '{\'proof\': {\'[["sleepy"], "impartial"]\': {\', \'[["attentive"], "sleepy"]\': {}}, \'label\': 0}',
  '{\'proof\': {\'[["weary", "rational"], "pessimistic"]\': {\',weary\': 1, \'[["precious", "distinct"], "rational"]\': {"["rational", "weary"], "precious"]\': {{\'rational\': 0}}}}, \'label\': 0}', 
  '{\'proof\': {\'[["precious"], "homely"]\': {\',precious\': 1}}}, \'label\': 1}', 
  "{'proof': {'intellectual': 1}, 'label': 1}",
  '{\'proof\': {\'[["elegant"], "sensible"]\': {\',[["beautiful", "perfect", "different"], "[["supportive"], "beautiful"]\': {{\'[[["adorable", "outrageous"], "supportive"]\': \'[["cruel", "adorable"], "amused"]\':{\'cruel\': 0}}}}}, \'label\': 0}', 
  '{\'proof\': {\'[["tense"], "good"]\': {\': {\',tense\': 0}}, \'label\': 0}}',
  '{\'proof\': {\'[["horrible"], "bad-tempered"]\': {\'][["shy", "distinct"], "horrible"]\':{\'shy\': 1, \'distinct\': 1}}}, \'label\': 1}', 
  "{'proof': {'dishonest': 1}, 'label': 1}", '{\'proof\': {\'[["sleepy", "thoughtful"], "thoughtless"]\': {\': {\',[["versatile", "clumsy", "selfish"], "sleepy"]\':{\'[[["frail", "hurt", "bright"], "versatile"]\': {{\'frail\': 1, \'[["thoughtless", "scared", "grail"], "hight"]\': {"["selfish", "versacious", "frail"], "[["hypocentristic", "beaut"], "selfishly"]\':}}}}}, \'label\': 0}'
 ]]




test_str = '{proof: {[["loving"], "unpleasant"]: {'loving': 1}}}, label: 1}'



for s in test_str.split():
    print(s)


