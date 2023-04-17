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

HOTPOTQA_PWS_UNLIMITED = '''Question: What is Leo Dicaprio's girlfriend's age to the power of 0.34? 
Plan: Find out the name of Leo Dicaprio's girlfriend.
#E1 = Google[name of Leo Dicaprio's girlfriend] 
Plan: Find out the age of Leo Dicaprio's girlfriend.
#E2 = Google[age of #E1]
Plan: Calculate her age to the power of 0.34.
#E3 = Calculator[#E2^0.34] 
'''