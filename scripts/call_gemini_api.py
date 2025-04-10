import base64
import os
from google import genai
from google.genai import types

GEMINI_API_KEY = "AIzaSyBZp-Qyp1jdNBGaqYy1WFgdn1NnNhjKPrY"



def extract_keyphrases(document: str) -> list:
    # Define the prompt for keyphrase extraction
    prompt = """
    ### Task: Extract the top 10 keyphrases from the given document that best summarize its main topics and concepts. 
            Keyphrases should be multi-word phrases where possible, focusing on noun phrases that capture the essence of the document. 
            Return the keyphrases as a numbered list, ensuring no duplicates and prioritizing the most relevant ideas.
            The keyphrases must be in the document, don't add anything that is not in the document.
            The response should be follow this example:

### Example 1:
- Document: "Climate change is affecting global temperatures, leading to more frequent heatwaves and droughts. Renewable energy sources like solar and wind power are becoming crucial for reducing carbon emissions. Governments are implementing policies to transition to green energy."
- Response:
# Climate change
# Global temperatures
# Heatwaves
# Droughts
# Renewable energy
# Solar power
# Wind power
# Carbon emissions
# Government policies
# Green energy

### Example 2:
- Document: "Machine learning algorithms are used in various applications, including image recognition and natural language processing. Deep learning models require large datasets for training, and neural networks are a key component."
- Response:
# Machine learning algorithms
# Image recognition
# Natural language processing
# Deep learning models
# Large datasets
# Neural networks
# Training data
# AI applications
# Model training
# Data requirements


    """

    client = genai.Client(api_key="AIzaSyBZp-Qyp1jdNBGaqYy1WFgdn1NnNhjKPrY")

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"{prompt}\n -Document: {document} \n -Response:\n"),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

    res = ""

    # Call the Gemini API to generate the keyphrases
    keyphrases = []
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        # print(chunk.text)
        res += chunk.text
    print(res)
    keyphrases = res.split("\n")
    keyphrases = [kw.replace("# ", "") for kw in keyphrases][:10]
    return [kw.lower() for kw in keyphrases]




def generate():
    # Example document to extract keyphrases from
    document = """Samson Mow Says Bitcoin Bear Trap: What’s Next at $82,516.97? We believe in full transparency with our readers . Some of our content includes affiliate links , and we may earn a commission through these partnerships . However , this potential compensation never influences our analysis , opinions , or reviews . Our editorial content is created independently of our marketing partnerships , and our ratings are based solely on our established evaluation criteria . Read More Bitcoin has slipped to $82,516.97 , down 1.55% in 24 hours , triggering concerns of a deeper correction . But according to Samson Mow  , this breakdown is a “bear trap”—a fakeout designed to flush out weak hands before a larger move higher. Mow remains firm on his $1 million BTC target , arguing the recent sell-off doesn’t reflect fundamentals . Bitcoin still holds a $1.64 trillion market cap , with 19.84 million BTC in circulation. While Mow sees upside ahead , charts tell a different story . BTC has broken below a symmetrical triangle , with the former support at $83,650 now acting as resistance. Bitcoin Technical Setup Signals Caution The recent breakdown from the symmetrical triangle pattern has turned $83,650 into a key resistance zone , stalling any immediate rebound . A bearish engulfing candle under this level signals continued selling pressure. Current Price : $82,516.97 - 24H Volume : $19.93B - Resistance Levels : $83,650 , $85,231 , $86,841 - Support Levels : $82,000 , $81,278 , $79,990 - 50 EMA : $85,231 (above current price) - RSI (14) : 27.63 (oversold) - The RSI remains oversold , but without bullish divergence , offering no clear sign of reversal . A breakdown through the triple bottom around $83,000 further weakens the structure , placing $81,278 and $79,990 in view . Volume near current levels is also subdued , reflecting a lack of strong buyer support. What Comes Next for Bitcoin? The key question is whether this is the trap Mow describes , or the start of a broader correction . Broader sentiment is mixed , with macroeconomic pressure and tight liquidity weighing on high-risk assets. A confirmed reclaim of $83,650 , followed by a breakout above the 50 EMA at $85,231 , would be an early signal of bullish recovery . Otherwise , continued failure at current levels risks a slide below $80,000. Key Signals to Monitor: Break and close above $83,650 - RSI divergence or recovery above 30 - Trading volume increase on bounce attempts - Support holding at $81,278 or $79,990 - Until these conditions are met , Bitcoin remains vulnerable . Whether Samson Mow’s call plays out will depend on how markets respond in the days ahead. BTC Bull: Earn Bitcoin Rewards with the Hottest Crypto Presale BTC Bull ($BTCBULL) is making waves as a community-driven token that automatically rewards holders with real Bitcoin when BTC hits key price milestones . Unlike traditional meme tokens , BTCBULL is built for long-term investors , offering real incentives through airdropped BTC rewards and staking opportunities. Staking & Passive Income Opportunities BTC Bull offers a high-yield staking program with an impressive 119% APY , allowing users to generate passive income . The staking pool has already attracted 882.5 million BTCBULL tokens , highlighting strong community participation. Latest Presale Updates: Current Presale Price: $0.002425 per BTCBULL - Total Raised: $4M / $4.5M target - With demand surging , this presale provides an opportunity to acquire BTCBULL at early-stage pricing before the next price increase.
"""

    keyphrases = extract_keyphrases(document)

    print("Top 10 Keyphrases:", keyphrases)

if __name__ == "__main__":
    generate()
