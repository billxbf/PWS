from prompts.wiki_prompt import *

HOTPOTQA_COT = '''What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Step 1: Identify the eastern sector of the Colorado Orogeny
The Colorado Orogeny refers to a series of mountain-building events that occurred in the Rocky Mountains, primarily in the present-day U.S. state of Colorado. The eastern sector of the Colorado Orogeny likely refers to the easternmost part of this mountain range, which includes the Front Range, Wet Mountains, and the Sangre de Cristo Mountains.
Step 2: Determine the elevation range of the eastern sector
The Front Range, Wet Mountains, and Sangre de Cristo Mountains have varying elevations. The Front Range has elevations ranging from around 5,000 feet to over 14,000 feet (1,500 meters to 4,300 meters). The Wet Mountains have elevations ranging from about 6,000 feet to over 12,000 feet (1,800 meters to 3,700 meters). The Sangre de Cristo Mountains have elevations ranging from approximately 7,000 feet to over 14,000 feet (2,100 meters to 4,300 meters).
Step 3: Combine the elevation ranges
Considering the elevation ranges of the three mountain ranges in the eastern sector of the Colorado Orogeny, the overall elevation range for the area extends from around 5,000 feet (1,500 meters) to over 14,000 feet (4,300 meters).
Answer: 5,000 feet to 4,000 feet
'''

HOTPOTQA_REACT = 'Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:\n(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.\n(3) Finish[answer], which returns the answer and finishes the task.\nHere are some examples.\n\nQuestion: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\nThought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.\nAction: Search[Colorado orogeny]\nObservation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.\nThought: It does not mention the eastern sector. So I need to look up eastern sector.\nAction: Lookup[eastern sector]\nObservation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.\nThought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.\nAction: Search[High Plains]\nObservation: High Plains refers to one of two distinct land regions\nThought: I need to instead search High Plains (United States).\nAction: Search[High Plains (United States)]\nObservation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\nThought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.\nAction: Finish[1,800 to 7,000 ft]\n\nQuestion: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?\nThought: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.\nAction: Search[Milhouse]\nObservation: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.\nThought: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".\nAction: Lookup[named after]\nObservation: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.\nThought: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.\nAction: Finish[Richard Nixon]\n\nQuestion: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\nThought: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.\nAction: Search[Adam Clayton Powell]\nObservation: Could not find [Adam Clayton Powell]. Similar: [’Adam Clayton Powell III’, ’Seventh Avenue (Manhattan)’, ’Adam Clayton Powell Jr. State Office Building’, ’Isabel Washington Powell’, ’Adam Powell’, ’Adam Clayton Powell (film)’, ’Giancarlo Esposito’].\nThought: To find the documentary, I can search Adam Clayton Powell (film).\nAction: Search[Adam Clayton Powell (film)]\nObservation: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.\nThought: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\nAction: Finish[The Saimaa Gesture]\n\nQuestion: What profession does Nicholas Ray and Elia Kazan have in common?\nThought: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.\nAction: Search[Nicholas Ray]\nObservation: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.\nThought: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.\nAction: Search[Elia Kazan]\nObservation: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.\nThought: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\nAction: Finish[director, screenwriter, actor]\n\nQuestion: Which magazine was started first Arthur’s Magazine or First for Women?\nThought: I need to search Arthur’s Magazine and First for Women, and find which was started first.\nAction: Search[Arthur’s Magazine]\nObservation: Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.\nThought: Arthur’s Magazine was started in 1844. I need to search First for Women next.\nAction: Search[First for Women]\nObservation: First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.\nThought: First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first.\nAction: Finish[Arthur’s Magazine]\n\nQuestion: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nThought: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\nAction: Search[Pavel Urysohn]\nObservation: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\nThought: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.\nAction: Search[Leonid Levin]\nObservation: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.\nThought: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.\nAction: Finish[yes]\n\n\nQuestion: {input}\n{agent_scratchpad}'


HOTPOTQA_PWS_BASE = '''Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Plan: Search for more information about Colorado orogeny.
#E1 = Wikipedia[Colorado orogeny]
Plan: Find out the area that eastern sector of the Colorado orogeny extends into.
#E2 = LLM[What is the name of the area that eastern sector of Colorado extends into? Given context: #E1]
Plan: Search for more information about the area.
#E3 = Wikipedia[#E2]
Plan: Find out the elevation range for the area.
#E4 = LLM[What is elevation range for the area #E2? Given context: #E3]

Question: Which magazine was started first Arthur's Magazine or First for Women?
Plan: Search for more information about Arthur's Magazine.
#E1 = Wikipedia[Arthur's Magazine]
Plan: Search for more information about First for Women.
#E2 = Wikipedia[First for Women]
Plan: Compare the two start dates and determine which magazine was started first.
#E3 = LLM[Which magazine was started first Arthur's Magazine or First for Women? Given context: #E1, #E2]
'''

HOTPOTQA_PWS_EXTRA = '''Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Plan: Search for more information about Colorado orogeny.
#E1 = Wikipedia[Colorado orogeny]
Plan: Find out the area that eastern sector of the Colorado orogeny extends into.
#E2 = LLM[What is the name of the area that eastern sector of Colorado extends into? Given context: #E1]
Plan: Search for more information about the area.
#E3 = Wikipedia[#E2]
Plan: Find out the elevation range for the area.
#E4 = LLM[What is elevation range for the area #E2? Given context: #E3]
Plan: Check on Google to see if the elevation range is correct.
#E5 = Google[Is the elevation range for #E2 #E4?]

Question: Which magazine was started first Arthur's Magazine or First for Women?
Plan: Search for more information about Arthur's Magazine.
#E1 = Wikipedia[Arthur's Magazine]
Plan: Search for more information about First for Women.
#E2 = Wikipedia[First for Women]
Plan: Compare the two start dates and determine which magazine was started first.
#E3 = LLM[Which magazine was started first Arthur's Magazine or First for Women? Given context: #E1, #E2]
Plan: Check on Google for additional check.
#E4 = Google[Is #E3 the magazine that was started first among Arthur's Magazine and First for Women?]
'''