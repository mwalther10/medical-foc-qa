class Prompts :
    def __init__(self) :
        self.summary_prompt = """
            Du bist ein hilfreicher medizinischer Assistent. Ein Patient beschreibt seine Situation. Fasse die Beschreibung des Patienten kurz und prägnant zusammen.
            {situation}
            """
        self.question_from_summary_prompt = """
            Formuliere aus folgender Situation eine prägnante Frage
            {situation}
        """
        self.split_prompt = """
        Du bist ein hilfreicher medizinischer Assistent in einem Forum für Patienten. 
        Ein Patient schildert seine Situation und stellt eine oder mehrere Frage am Ende der Beschreibung. 
        Beantworte die Frage(n) des Patienten kurz und prägnant. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt, versuche nicht, eine Antwort zu erfinden.
        Die Situation des Patienten ist:
        {situation}
        Die Frage lautet:
        {questions}
        """
        self.plain_prompt = """
        Du bist ein hilfreicher medizinischer Assistent in einem Forum für Patienten. 
        Ein Patient schildert seine Situation und stellt eine oder mehrere Frage am Ende der Beschreibung. Beantworte die Frage(n) des Patienten kurz und prägnant. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt, versuche nicht, eine Antwort zu erfinden.
        Die Frage des Patienten lautet:
        {text}
        """
        self.judge_prompt = """
        Bewerte, ob die beiden folgenden medizinischen Statements die gleichen Informationen enthalten. Falls sie die gleichen Informationen enthalten, wähle "Ja", ansonsten wähle "Nein".

        Statement 1 {statement1}

        Statement 2 {statement2}

        Hilfreiche Antwort:"""

        self.extract_facts_prompt = """
        Du bist ein hilfreicher medizinsicher Assistent. Extrahiere die medizinischen Fakten aus dem folgenden Text und gebe sie in Stichpunkten zurück.

        Text {text}

        Hilfreiche Antwort:"""

        self.rag_prompt = """
        Du bist ein hilfreicher medizinischer Assistent in einem Forum für Patienten. Beantworte die Frage am Ende des Textes immer anhand der folgenden Informationen.
        Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt, versuche nicht, eine Antwort zu erfinden. Verwende leichte Sprache.

        {context}

        Frage: {question}

        Hilfreiche Antwort:"""

        self.similar_statements = """
        Welche Statements aus den folgenden zwei Listen von Statements sind gleich?.

        Liste 1: {list1}

        Liste 2: {list2}

        Gleiche Statements:"""

    
    def get_plain_prompt(self,query) :
        return self.plain_prompt.format(text=query)
    
    def get_split_prompt(self,situation,questions) :
        return self.split_prompt.format(situation=situation,questions=questions)
    
    def get_summary_prompt(self,situation) :
        return self.summary_prompt.format(situation=situation)
    
    def get_question_from_summary_prompt(self,situation) :
        return self.question_from_summary_prompt.format(situation=situation)
    
    def get_rag_prompt(self, context, question):
        return self.rag_prompt.format(context=context, question=question)
    
    def get_judge_prompt(self, statement1, statement2):
        return self.judge_prompt.format(statement1=statement1, statement2=statement2)
    
    def get_extract_facts_prompt(self, text):
        return self.extract_facts_prompt.format(text=text)
    
    def get_similar_statements(self, list1, list2):
        return self.similar_statements.format(list1=list1, list2=list2)
    

# first summary prompt for first experiments
# self.summary_prompt = """
#             Fasse folgende Situation zusammen und gebe nur Stichpunkte aus, die zur Zusammenfassung der Situation beitragen:
#             {situation}
#             """