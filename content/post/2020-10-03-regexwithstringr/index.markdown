---
title: '[R] stringr 패키지를 사용한 정규표현식'
author: JDW
date: '2020-10-03'
slug: regexwithstringr
categories:
  - R
tags:
  - R
  - 텍스트
  - 전처리
  - 정규표현식
  - stringr
subtitle: ''
summary: ''
authors: []
lastmod: '2020-10-03T17:53:43+09:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: yes
projects: []
output:
  blogdown::html_page:
    toc: TRUE
    toc_depth: 4
---



<img src="images/stringr_logo.png" alt="" width="50%"/>


&nbsp;이전 장에서 R에서 사용되는 정규표현식에 대해 알아보았었는데요. 오늘은 정규표현식을 텍스트 데이터를 처리할때 사용하는 패키지인 __`stringr`__ 과 함께 사용하는 방법에 대해 알아보도록 하겠습니다. 

# stringr 
&nbsp; __`stringr`__ 패키지는 Rstudio의 Hadley Wickham이 개발한 패키지입니다. __`dplyr`__ , __`ggplot2`__ 등과 같이 데이터를 깔끔한(tidy) 방식으로 처리하는것을 지향하는 **`tidyverse`** 패키지군에 속해 있으며, 그 중 __`stringr`__ 은 문자열 처리에 특화된 패키지입니다. 'string'의 준말인 `str_`이 함수 머릿말에 항상 등장하는 것이 특징입니다.

&nbsp; __`stringr`__ 패키지의 주요 기능을 아래와 같이 분류하였고 각 경우에 맞는 상황을 가정하여 알아보도록 하겠습니다. 

[1. 특정 패턴과 **'일치'**](#일치) <br>
[2. 특정 패턴의 **'카운팅'**](#카운팅) <br>
[3. 특정 패턴이 __'포함'__ 된 경우](#포함) <br>
[4. 특정 패턴의 __'위치'__ 를 반환](#위치) <br>
[5. 특정 패턴을 새로운 것으로 **'바꾸는'** 경우](#바꾸기) <br>
[6. 특정 패턴을 기준으로 **'나누는'** 경우](#나누기)<br>

## 예제 문장 
&nbsp;연습용으로 사용할 문장은 stringr 패키지에 내장되어 있는 `sentence` 데이터를 사용하겠습니다. `sentence`는 총 1000문장으로 구성되어 있는데, 문장 전체를 사용할 필요가 없으므로 10문장 정도만을 추려서 사용하도록 하겠습니다. 


```r
regex_sentences <- stringr::sentences[1:10] # 예제용 문장 생성

regex_sentences
#>  [1] "The birch canoe slid on the smooth planks." 
#>  [2] "Glue the sheet to the dark blue background."
#>  [3] "It's easy to tell the depth of a well."     
#>  [4] "These days a chicken leg is a rare dish."   
#>  [5] "Rice is often served in round bowls."       
#>  [6] "The juice of lemons makes fine punch."      
#>  [7] "The box was thrown beside the parked truck."
#>  [8] "The hogs were fed chopped corn and garbage."
#>  [9] "Four hours of steady work faced us."        
#> [10] "Large size in stockings is hard to sell."
```

### 일치
- &nbsp;`str_detect()`함수는 특정 문자 혹은 패턴(정규표현식)과 일치하는 경우를 찾을때 사용하며, 실행 결과로서 boolean(참, 거짓)을 반환합니다.  

```r
str_detect(regex_sentences, '[aeiou]') # 알파벳 모음 'a' or 'e' or 'i' or 'o' or 'u'가 있을 경우 TRUE
#>  [1] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
```


```r
str_detect(regex_sentences, '[easy]')  # 알파벳 'e' or 'a' or 's' or 'y'가 있을 경우 TRUE 
#>  [1] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
```


```r
str_detect(regex_sentences, '(easy)')  # 알파벳 'easy'가 있을 경우 TRUE 
#>  [1] FALSE FALSE  TRUE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
```


- &nbsp;`str_extract()`함수는 특정 문자 혹은 패턴(정규표현식)과 일치하는 경우, 그것을 반환합니다. 

```r
str_extract(regex_sentences, '[aeiou]') # 'a' or 'e' or 'i' or 'o' or 'u' 글자가 존재하면 그것을 반환 
#>  [1] "e" "u" "e" "e" "i" "e" "e" "e" "o" "a"
```


```r
str_extract(regex_sentences, '[:punct:]') # 문장부호가 존재하면 그것을 반환 
#>  [1] "." "." "'" "." "." "." "." "." "." "."
```


&nbsp;함수는 조건에 일치하는 결과를 반환하였는데요. 자세히 보면 조건에 일치한 것이 여러개 있음에도 불구하고 오직 최초 값만을 반환하는 것을 볼 수 있습니다. 그렇다면 조건에 일치하는 모든 값을 찾고자 할 땐 어떻게 해야할까요? 바로 `str_extract_all()` 함수를 사용하면 됩니다. __`stringr`__ 패키지 함수 중 몇몇의 함수는 함수 말미에 `_all`이 붙는 함수들이 있습니다. 이들 함수들은 조건에 맞는 모든 값을 반환하는 특징이 있습니다.

```r
str_extract_all(regex_sentences, '[aeiou]')
#> [[1]]
#>  [1] "e" "i" "a" "o" "e" "i" "o" "e" "o" "o" "a"
#> 
#> [[2]]
#>  [1] "u" "e" "e" "e" "e" "o" "e" "a" "u" "e" "a" "o" "u"
#> 
#> [[3]]
#> [1] "e" "a" "o" "e" "e" "e" "o" "a" "e"
#> 
#> [[4]]
#>  [1] "e" "e" "a" "a" "i" "e" "e" "i" "a" "a" "e" "i"
#> 
#> [[5]]
#>  [1] "i" "e" "i" "o" "e" "e" "e" "i" "o" "u" "o"
#> 
#> [[6]]
#>  [1] "e" "u" "i" "e" "o" "e" "o" "a" "e" "i" "e" "u"
#> 
#> [[7]]
#>  [1] "e" "o" "a" "o" "e" "i" "e" "e" "a" "e" "u"
#> 
#> [[8]]
#>  [1] "e" "o" "e" "e" "e" "o" "e" "o" "a" "a" "a" "e"
#> 
#> [[9]]
#>  [1] "o" "u" "o" "u" "o" "e" "a" "o" "a" "e" "u"
#> 
#> [[10]]
#>  [1] "a" "e" "i" "e" "i" "o" "i" "i" "a" "o" "e"
```


```r
str_extract_all(regex_sentences, '[:punct:]')
#> [[1]]
#> [1] "."
#> 
#> [[2]]
#> [1] "."
#> 
#> [[3]]
#> [1] "'" "."
#> 
#> [[4]]
#> [1] "."
#> 
#> [[5]]
#> [1] "."
#> 
#> [[6]]
#> [1] "."
#> 
#> [[7]]
#> [1] "."
#> 
#> [[8]]
#> [1] "."
#> 
#> [[9]]
#> [1] "."
#> 
#> [[10]]
#> [1] "."
```

&nbsp;`str_extract_all()` 함수의 결과, 조건에 부합하는 모든 값이 list 형으로 반환된 것을 볼 수 있습니다. 

### 카운팅 
- &nbsp;`str_count()`함수는 특정 문자 혹은 패턴(정규표현식)과 일치하는 경우를 계산합니다. 

```r
str_count(regex_sentences, '[aeiou]') # 'a' or 'e' or 'i' or 'o' or 'u' 문자의 개수를 반환 
#>  [1] 11 13  9 12 11 12 11 12 11 11
```


```r
str_count(regex_sentences, '^(The)') # 시작이 'The'인 경우를 셈 
#>  [1] 1 0 0 1 0 1 1 1 0 0
```


### 포함
- &nbsp;`str_subset()`함수는 특정 문자 혹은 패턴(정규표현식)이 포함된 경우를 반환합니다. 

```r
str_subset(regex_sentences, '[aeiou]') # 'a' or 'e' or 'i' or 'o' or 'u' 문자가 포함된 경우를 반환 
#>  [1] "The birch canoe slid on the smooth planks." 
#>  [2] "Glue the sheet to the dark blue background."
#>  [3] "It's easy to tell the depth of a well."     
#>  [4] "These days a chicken leg is a rare dish."   
#>  [5] "Rice is often served in round bowls."       
#>  [6] "The juice of lemons makes fine punch."      
#>  [7] "The box was thrown beside the parked truck."
#>  [8] "The hogs were fed chopped corn and garbage."
#>  [9] "Four hours of steady work faced us."        
#> [10] "Large size in stockings is hard to sell."
```


```r
str_subset(regex_sentences, '^(The)')  # 시작이 'The'인 문장을 반환
#> [1] "The birch canoe slid on the smooth planks." 
#> [2] "These days a chicken leg is a rare dish."   
#> [3] "The juice of lemons makes fine punch."      
#> [4] "The box was thrown beside the parked truck."
#> [5] "The hogs were fed chopped corn and garbage."
```


```r
str_subset(regex_sentences, 'p[:alpha:]{1,}\\.$') # 'p'뒤에 알파벳이 한번 이상 나오면서 마침표로 끝나는 경우 
#> [1] "The birch canoe slid on the smooth planks."
#> [2] "The juice of lemons makes fine punch."
```


### 위치
- &nbsp;`str_locate()`함수는 특정 문자 혹은 패턴(정규표현식)에 맞는 경우의 시작 지점과 끝 지점을 반환합니다. R에 기본적으로 내장되어 있는 영어 소문자 데이터인 `letters`를 통해 알아보겠습니다. 

```r
letters
#>  [1] "a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" "m" "n" "o" "p" "q" "r" "s"
#> [20] "t" "u" "v" "w" "x" "y" "z"
```

&nbsp;패턴의 위치를 찾기 위해선 하나로 묶인 데이터가 필요하므로 `str_c()` 함수를 사용하여 하나의 벡터로 묶었습니다. (`str_c()`함수는 __`stringr`__패키지의 문자를 결합하는 함수. R의 내장함수인 `paste()`와 동일하게 동작)

```r
letters <- str_c(letters, collapse = '');

letters # 하나의 벡터값으로 변환 
#> [1] "abcdefghijklmnopqrstuvwxyz"
```


```r
str_locate(letters, '[a-z]+') # 'a'부터 'z' 까지의 문자가 한개 이상 나오는 경우의 시작과 끝을 반환 
#>      start end
#> [1,]     1  26
```


```r
str_locate(letters, '^[a][a-z]+[z]$') # 시작이 'a'이며, 바로 뒤에 'a' 부터 'z'까지의 문자 중 하나가 한번 이상 나오며, 끝이 'z'인 경우의 시작 지점과 끝 지점을 반환.
#>      start end
#> [1,]     1  26
```


```r
str_locate(letters, '^[a-z][a-z]+$') # 시작이 'a' 부터 'z' 까지의 문자 중 하나이며, 끝이 'a' 부터 'z' 까지의 문자 중 하나인 경우의 시작 지점과 끝 지점.
#>      start end
#> [1,]     1  26
```


```r
str_locate(letters, '(abcde)') # 'abcde'의 시작 지점과 끝 지점을 반환 
#>      start end
#> [1,]     1   5
```


```r
str_locate(letters, '(xyz)') # 'xyz'의 시작 지점과 끝 지점을 반환  
#>      start end
#> [1,]    24  26
```


&nbsp;`str_locate_all()`함수를 통해 조건에 맞는 모든 값의 위치를 추출할 수 있습니다. 

```r
str_locate(letters, '[aeiou][^aeiou]+') # 'a' or 'e' or 'i' or 'o' or 'u'문자 뒤에 'a' or 'e' or 'i' or 'o' or 'u' 문자가 아닌 문자가 하나 이상 나오는 최초 경우의 시작 지점과 끝 지점을 반환 
#>      start end
#> [1,]     1   4
```


```r
str_locate_all(letters, '[aeiou][^aeiou]+') # 'a' or 'e' or 'i' or 'o' or 'u'문자 뒤에 'a' or 'e' or 'i' or 'o' or 'u' 문자가 아닌 문자가 하나 이상 나오는 모든 경우의 시작 지점과 끝 지점을 반환 
#> [[1]]
#>      start end
#> [1,]     1   4
#> [2,]     5   8
#> [3,]     9  14
#> [4,]    15  20
#> [5,]    21  26
```


### 바꾸기 
- &nbsp;`str_replace()`는 특정 문자 혹은 패턴(정규표현식)에 맞는 경우를 새로운 문자로 바꾸는 함수입니다. 

```r
str_replace(regex_sentences, 'k', "'K'") # 최초로 매칭되는 문자 k를 'K'로 변환  
#>  [1] "The birch canoe slid on the smooth plan'K's." 
#>  [2] "Glue the sheet to the dar'K' blue background."
#>  [3] "It's easy to tell the depth of a well."       
#>  [4] "These days a chic'K'en leg is a rare dish."   
#>  [5] "Rice is often served in round bowls."         
#>  [6] "The juice of lemons ma'K'es fine punch."      
#>  [7] "The box was thrown beside the par'K'ed truck."
#>  [8] "The hogs were fed chopped corn and garbage."  
#>  [9] "Four hours of steady wor'K' faced us."        
#> [10] "Large size in stoc'K'ings is hard to sell."
```


```r
str_replace(regex_sentences, '\\.', '\\?') # 최초로 매칭되는 마침표를 물음표로 변환 
#>  [1] "The birch canoe slid on the smooth planks?" 
#>  [2] "Glue the sheet to the dark blue background?"
#>  [3] "It's easy to tell the depth of a well?"     
#>  [4] "These days a chicken leg is a rare dish?"   
#>  [5] "Rice is often served in round bowls?"       
#>  [6] "The juice of lemons makes fine punch?"      
#>  [7] "The box was thrown beside the parked truck?"
#>  [8] "The hogs were fed chopped corn and garbage?"
#>  [9] "Four hours of steady work faced us?"        
#> [10] "Large size in stockings is hard to sell?"
```

&nbsp;마찬가지로 `str_replace_all()`함수를 통해 조건에 맞는 모든 값을 바꿀 수 있습니다. 

```r
str_replace(regex_sentences, '[:alpha:]{1,}', 'word') # 최초로 매칭되는 알파뱃이 한개 이상인 경우를 'word'로 변환 
#>  [1] "word birch canoe slid on the smooth planks." 
#>  [2] "word the sheet to the dark blue background." 
#>  [3] "word's easy to tell the depth of a well."    
#>  [4] "word days a chicken leg is a rare dish."     
#>  [5] "word is often served in round bowls."        
#>  [6] "word juice of lemons makes fine punch."      
#>  [7] "word box was thrown beside the parked truck."
#>  [8] "word hogs were fed chopped corn and garbage."
#>  [9] "word hours of steady work faced us."         
#> [10] "word size in stockings is hard to sell."
```


```r
str_replace_all(regex_sentences, '[:alpha:]{1,}', 'word') # 알파뱃이 한개 이상인 모든 값을 'word'라는 단어로 변환 
#>  [1] "word word word word word word word word."          
#>  [2] "word word word word word word word word."          
#>  [3] "word'word word word word word word word word word."
#>  [4] "word word word word word word word word word."     
#>  [5] "word word word word word word word."               
#>  [6] "word word word word word word word."               
#>  [7] "word word word word word word word word."          
#>  [8] "word word word word word word word word."          
#>  [9] "word word word word word word word."               
#> [10] "word word word word word word word word."
```


```r
str_replace(regex_sentences, '[:space:]', '') # 최초로 매칭되는 띄어쓰기를 공백으로 변환 
#>  [1] "Thebirch canoe slid on the smooth planks." 
#>  [2] "Gluethe sheet to the dark blue background."
#>  [3] "It'seasy to tell the depth of a well."     
#>  [4] "Thesedays a chicken leg is a rare dish."   
#>  [5] "Riceis often served in round bowls."       
#>  [6] "Thejuice of lemons makes fine punch."      
#>  [7] "Thebox was thrown beside the parked truck."
#>  [8] "Thehogs were fed chopped corn and garbage."
#>  [9] "Fourhours of steady work faced us."        
#> [10] "Largesize in stockings is hard to sell."
```


```r
str_replace_all(regex_sentences, '[:space:]', '') # 매칭되는 모든 띄어쓰기를 공백으로 변환 
#>  [1] "Thebirchcanoeslidonthesmoothplanks." 
#>  [2] "Gluethesheettothedarkbluebackground."
#>  [3] "It'seasytotellthedepthofawell."      
#>  [4] "Thesedaysachickenlegisararedish."    
#>  [5] "Riceisoftenservedinroundbowls."      
#>  [6] "Thejuiceoflemonsmakesfinepunch."     
#>  [7] "Theboxwasthrownbesidetheparkedtruck."
#>  [8] "Thehogswerefedchoppedcornandgarbage."
#>  [9] "Fourhoursofsteadyworkfacedus."       
#> [10] "Largesizeinstockingsishardtosell."
```


### 나누기
- &nbsp;`str_split()`는 특정 문자 혹은 패턴(정규표현식)을 기준으로 값을 나누는 함수입니다. 이때 기준이된 문자 혹은 패턴은 제거됩니다. 

```r
str_split(regex_sentences, '[:upper:]') # 대문자를 기준으로 분절 
#> [[1]]
#> [1] ""                                         
#> [2] "he birch canoe slid on the smooth planks."
#> 
#> [[2]]
#> [1] ""                                          
#> [2] "lue the sheet to the dark blue background."
#> 
#> [[3]]
#> [1] ""                                     
#> [2] "t's easy to tell the depth of a well."
#> 
#> [[4]]
#> [1] ""                                       
#> [2] "hese days a chicken leg is a rare dish."
#> 
#> [[5]]
#> [1] ""                                    "ice is often served in round bowls."
#> 
#> [[6]]
#> [1] ""                                    
#> [2] "he juice of lemons makes fine punch."
#> 
#> [[7]]
#> [1] ""                                          
#> [2] "he box was thrown beside the parked truck."
#> 
#> [[8]]
#> [1] ""                                          
#> [2] "he hogs were fed chopped corn and garbage."
#> 
#> [[9]]
#> [1] ""                                   "our hours of steady work faced us."
#> 
#> [[10]]
#> [1] ""                                       
#> [2] "arge size in stockings is hard to sell."
```


```r
str_split(regex_sentences, '\\b') # 문자의 바운더리를 기준으로 분절 
#> [[1]]
#>  [1] ""       "The"    " "      "birch"  " "      "canoe"  " "      "slid"  
#>  [9] " "      "on"     " "      "the"    " "      "smooth" " "      "planks"
#> [17] "."     
#> 
#> [[2]]
#>  [1] ""           "Glue"       " "          "the"        " "         
#>  [6] "sheet"      " "          "to"         " "          "the"       
#> [11] " "          "dark"       " "          "blue"       " "         
#> [16] "background" "."         
#> 
#> [[3]]
#>  [1] ""      "It"    "'"     "s"     " "     "easy"  " "     "to"    " "    
#> [10] "tell"  " "     "the"   " "     "depth" " "     "of"    " "     "a"    
#> [19] " "     "well"  "."    
#> 
#> [[4]]
#>  [1] ""        "These"   " "       "days"    " "       "a"       " "      
#>  [8] "chicken" " "       "leg"     " "       "is"      " "       "a"      
#> [15] " "       "rare"    " "       "dish"    "."      
#> 
#> [[5]]
#>  [1] ""       "Rice"   " "      "is"     " "      "often"  " "      "served"
#>  [9] " "      "in"     " "      "round"  " "      "bowls"  "."     
#> 
#> [[6]]
#>  [1] ""       "The"    " "      "juice"  " "      "of"     " "      "lemons"
#>  [9] " "      "makes"  " "      "fine"   " "      "punch"  "."     
#> 
#> [[7]]
#>  [1] ""       "The"    " "      "box"    " "      "was"    " "      "thrown"
#>  [9] " "      "beside" " "      "the"    " "      "parked" " "      "truck" 
#> [17] "."     
#> 
#> [[8]]
#>  [1] ""        "The"     " "       "hogs"    " "       "were"    " "      
#>  [8] "fed"     " "       "chopped" " "       "corn"    " "       "and"    
#> [15] " "       "garbage" "."      
#> 
#> [[9]]
#>  [1] ""       "Four"   " "      "hours"  " "      "of"     " "      "steady"
#>  [9] " "      "work"   " "      "faced"  " "      "us"     "."     
#> 
#> [[10]]
#>  [1] ""          "Large"     " "         "size"      " "         "in"       
#>  [7] " "         "stockings" " "         "is"        " "         "hard"     
#> [13] " "         "to"        " "         "sell"      "."
```


```r
str_split(regex_sentences, '\\s') # 띄어쓰기를 기준으로 분절 
#> [[1]]
#> [1] "The"     "birch"   "canoe"   "slid"    "on"      "the"     "smooth" 
#> [8] "planks."
#> 
#> [[2]]
#> [1] "Glue"        "the"         "sheet"       "to"          "the"        
#> [6] "dark"        "blue"        "background."
#> 
#> [[3]]
#> [1] "It's"  "easy"  "to"    "tell"  "the"   "depth" "of"    "a"     "well."
#> 
#> [[4]]
#> [1] "These"   "days"    "a"       "chicken" "leg"     "is"      "a"      
#> [8] "rare"    "dish."  
#> 
#> [[5]]
#> [1] "Rice"   "is"     "often"  "served" "in"     "round"  "bowls."
#> 
#> [[6]]
#> [1] "The"    "juice"  "of"     "lemons" "makes"  "fine"   "punch."
#> 
#> [[7]]
#> [1] "The"    "box"    "was"    "thrown" "beside" "the"    "parked" "truck."
#> 
#> [[8]]
#> [1] "The"      "hogs"     "were"     "fed"      "chopped"  "corn"     "and"     
#> [8] "garbage."
#> 
#> [[9]]
#> [1] "Four"   "hours"  "of"     "steady" "work"   "faced"  "us."   
#> 
#> [[10]]
#> [1] "Large"     "size"      "in"        "stockings" "is"        "hard"     
#> [7] "to"        "sell."
```

___

# 참고자료 

- [Rstudio Regular Expression Cheatsheet](https://github.com/rstudio/cheatsheets/blob/master/regex.pdf)

- [Rstudio Stringr Cheatsheets](https://github.com/rstudio/cheatsheets/blob/master/strings.pdf)

- [R을 활용한 데이터과학 - 문자열](https://sulgik.github.io/r4ds/strings.html)
