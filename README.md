AhoCorasickDoubleArrayTrie
============

An extremely fast implementation of Aho Corasick algorithm based on Double Array Trie structure. Its speed is 1.7 to 4.5 times of naive implementations, perhaps it's the fastest implementation so far ;-)

Introduction
------------
You may heard that Aho-Corasick algorithm is fast for parsing text with a huge dictionary, for example:
* looking for certain words in texts in order to URL link or emphasize them
* adding semantics to plain text
* checking against a dictionary to see if syntactic errors were made

But most implementation use a `TreeMap<Character, State>` to store the success function, which costs `O(n*ln(t))` time, `n` is the length of text, and `t` is the largest amount of a word's common suffixes, absolutely `t > 2`. The others used a `HashMap`, which wasted too much memory, and still remained slow.

I improve it by replace the `XXXMap` to a Double Array Trie, whose time complexity is just `O(n)`, and has a perfect balance of time and memory. Yes, its speed is not related to the length or language or common suffix of the words of a dictionary.

Usage
-----
Setting up the `AhoCorasickDoubleArrayTrie` is a piece of cake:
```java
        // Collect test data set
        Map<String, String> map = new TreeMap<String, String>();
        String[] keyArray = new String[]
                {
                        "hers",
                        "his",
                        "she",
                        "he"
                };
        for (String key : keyArray)
        {
            map.put(key, key);
        }
        // Build an AhoCorasickDoubleArrayTrie
        AhoCorasickDoubleArrayTrie<String> act = new AhoCorasickDoubleArrayTrie<String>();
        act.build(map);
        // Test it
        final String text = "uhers";
        List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> segmentList = act.parseText(text);
```

Of course, there remains many usefull method to be discoverd, feel free to try:
* Use a `Map<String, Object>` to assign a Object as value to a keyword.
* Store the `AhoCorasickDoubleArrayTrie` to disk by calling `save` method.
* Restore the `AhoCorasickDoubleArrayTrie` from disk by calling `load` method.

In normal situations you probably do not need a huge segmentList, then please try this:

```java
        act.parseText(text, new AhoCorasickDoubleArrayTrie.IHit<String>()
        {
            @Override
            public void hit(int begin, int end, String value)
            {
                System.out.printf("[%d:%d]=%s\n", begin, end, value);
            }
        });
```

Comparison
-----
I compared my AhoCorasickDoubleArrayTrie with robert-bor's aho-corasick, ACDAT represents for AhoCorasickDoubleArrayTrie and Naive repesents for aho-corasick, the result is :
```
Parsing document which has 3409283 characters, with a dictionary of 127142 words.
               	Naive          	ACDAT          
time           	589            	333            
char/s         	5788256.37     	10238087.09    
rate           	1.00           	1.77           
===========================================================================
Parsing document which has 1290573 characters, with a dictionary of 146047 words.
               	Naive          	ACDAT          
time           	316            	70             
char/s         	4084091.77     	18436757.14    
rate           	1.00           	4.51           
===========================================================================
```



License
-------
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.