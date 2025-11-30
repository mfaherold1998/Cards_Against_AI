
# PROMPTS[lang][prompt_name] => prompt text

PROMPTS = {
    "EN": 
    {
       
        "prompt_player" : '''
                        Hi! Let\'s play a round of Cards Against Humanity.
                        You will be the player.
                        Card Zar personality: {character_description}
                        The Black card selected is: {black_card_text}.
                        The White cards available are: {white_cards_options}.
                        Which card do you choose? Remember that the goal is to choose the funniest one and will be judged by the zar card.
                        As a response, give me just the id of the card you choose in a format: "ID: card_id".
                        ''',
        "prompt_judge" : '''
                        Hi! Let\'s play a round of Cards Against Humanity.
                        You will be the Card Zar.
                        Card Zar personality: {character_description}
                        The Black card selected is: {black_card_text}.
                        The White cards to judge are: {white_cards_options}.
                        Which card do you choose? Remember that the goal is to choose the funniest one and will be judged by the zar card.
                        As a response, give me just the id of the card you choose in a format: "ID: card_id".
                        '''        
    },

    "IT": {},
    "ES": {}
}