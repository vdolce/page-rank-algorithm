import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    print(corpus)

    # PageRank estimation by sampling
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    
    # all probabilities must sum up to 1
    print("sum probabilities = " + str(sum(ranks.values())))

    # # PageRank estimation by iteration
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    # all probabilities must sum up to 1
    print("sum probabilities = " + str(sum(ranks.values())))

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}
            

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    
    linked_pages = corpus[page]
    all_pages_probability = (1 - damping_factor)/len(corpus) 

    if len(linked_pages) == 0:
        result = dict.fromkeys(corpus.keys(), 1/len(corpus))
        return result
    else:
        result = dict.fromkeys(corpus.keys(),0)
        linked_pages_probabilities = damping_factor/ len(linked_pages)
    
    for page in result.keys():
        if page in linked_pages:
            result[page] +=  linked_pages_probabilities
        result[page] += all_pages_probability 

    return result
    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    visited_pages_counter = dict.fromkeys(corpus.keys(),0)

    # first random page
    next_page = random.choice(list(corpus.keys()))

    for i in range(SAMPLES):
        sample = transition_model(corpus, next_page, damping_factor)
        next_page = random.choices(list(sample.keys()), k=1, weights=list(sample.values()))[0]
        visited_pages_counter[next_page] += 1

    result = dict.fromkeys(visited_pages_counter)

    # calculate probability based on the number of times that each page's been visited
    for key in result:
        result[key] = visited_pages_counter[key]/n

    return result
    raise NotImplementedError

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # dictionary that contains the list of pages that have a link to the key page
    # key = single page 
    # values = list of pages that have a link to the key
    links_to_page = dict.fromkeys(corpus.keys(), [] )

    for key1 in links_to_page.keys():
        pages_list = []
        for key2 in corpus.keys():
            if key1 in corpus[key2]:
                pages_list.append(key2)

        links_to_page[key1] = pages_list

    print(links_to_page)

    ACCURACY = 0.001

    # initialize the result dict -> all pages have the same probability - step t
    result = dict.fromkeys(corpus.keys(), 1/len(corpus) )
    # initialize the result_next dict -> step t + 1
    next_result = dict.fromkeys(corpus.keys(), 0 )

    # used to break the while loop when probabilities converge based on accuracy
    converge_probabilities = dict.fromkeys(corpus.keys(), 0 )

    # With probability d, the surfer followed a link from a page i to page p.
    i=0
    new_value_part1 = (1-damping_factor)/len(corpus)

    while sum(converge_probabilities.values()) < len(corpus):
        for key in list(corpus.keys()):
            
            new_value_part2 = 0
            for l in list(links_to_page[key]):
                if len(corpus[l]) > 0:             
                    new_value_part2 += damping_factor * ( result[l] / len(corpus[l]))

            # update the result dict at t+1 step
            next_result[key] = new_value_part1 + new_value_part2

            if(abs(next_result[key] - result[key]) < ACCURACY):
                converge_probabilities[key] = 1
            else:
                converge_probabilities[key] = 0
        
        result = next_result.copy()

    return result

    raise NotImplementedError


if __name__ == "__main__":
    main()
