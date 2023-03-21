import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia(
        language='fr',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

p_wiki = wiki_wiki.page("Pandémie_de_Covid-19")
print(p_wiki.text)
