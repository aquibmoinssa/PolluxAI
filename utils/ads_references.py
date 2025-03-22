from astroquery.nasa_ads import ADS
import re
import logging
import os

def extract_keywords_with_gpt(context, client, max_tokens=100, temperature=0.3):
    
    keyword_prompt = f"Extract 3 most important scientific keywords from the following user query:\n\n{context}"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in identifying key scientific terms and concepts."},
            {"role": "user", "content": keyword_prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    extracted_keywords = response.choices[0].message.content.strip()
    
    cleaned_keywords = re.sub(r'\d+\.\s*', '', extracted_keywords)

    keywords_list = [kw.strip() for kw in cleaned_keywords.split("\n") if kw.strip()]
    
    return keywords_list

def fetch_nasa_ads_references(ads_query):
    """Fetch relevant NASA ADS papers and format them for readability."""
    ADS.TOKEN = os.getenv('ADS_API_KEY')
    try:
        # Query NASA ADS for relevant papers
        papers = ADS.query_simple(ads_query)

        if not papers or len(papers) == 0:
            return [("No results found", "N/A", "N/A", "N/A", "N/A", "N/A")]

        # Include authors in the references
        references = []
        for paper in papers[:5]:  # Limit to 5 references
            title = paper.get('title', ['Title not available'])[0]
            abstract = paper.get('abstract', 'Abstract not available')
            authors = ", ".join(paper.get('author', [])[:3]) + (" et al." if len(paper.get('author', [])) > 3 else "")
            bibcode = paper.get('bibcode', 'N/A')
            pub = paper.get('pub', 'Unknown Journal')
            pubdate = paper.get('pubdate', 'Unknown Date')

            references.append((title, abstract, authors, bibcode, pub, pubdate))

        return references

    except Exception as e:
        logging.error(f"Error fetching ADS references: {str(e)}")
        return [("Error fetching references", "See logs for details", "N/A", "N/A", "N/A", "N/A")]

