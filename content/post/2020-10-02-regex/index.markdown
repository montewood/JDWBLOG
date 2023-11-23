---
title: '[R] 정규표현식 활용한 텍스트 데이터 다루기'
author: JDW
date: '2020-10-02'
slug: regex
categories:
  - R
tags:
  - R
  - 텍스트
  - 전처리
  - 정규표현식
subtitle: ''
summary: ''
authors: []
lastmod: '2020-10-02T14:00:32+09:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: yes
projects: []
output:
  blogdown::html_page:
    toc: TRUE
    toc_depth: 3
---

<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<link href="{{< blogdown/postref >}}index_files/str_view/str_view.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/str_view-binding/str_view.js"></script>

![](images/regular-expression.gif)

 문자형 데이터를 다루는데는 수 많은 기법들이 있습니다. R에서 제공하는 기본 함수들을 통해 처리하기도 하고, 텍스트처리에 특화된 패키지함수를 사용하여 원하는 처리를 할 수 있는데요. 오늘은 그 중에 텍스트데이터를 다루는데 효과적으로 사용되는 **정규표현식**을 사용하는 법에 대해 알아보도록 하겠습니다.

# 정규표현식

 **정규표현식**(regular expression)은 특정한 규칙을 가진 문자열의 집합을 표현하는데 사용하는 형식 언어입니다.
프로그래밍에서 조건문과 반복문을 통해 원하는 결과값을 도출해내는 것처럼, 텍스트데이터에서 조건과 패턴에 맞는 문자열을 찾거나 추출할때 정규식을 사용합니다. 정규표현식에는 특별한 의미를 지니고 있는 *메타문자* 와 문자 그대로의 의미를 지닌 *일반문자* 로 구성되어 있습니다.

## 메타문자

 메타문자는 정규표현식에서 특별한 의미를 갖는 문자 기호입니다. 키보드에서 알파벳과 숫자를 제외한 특수 기호중에서 미리 정의한 기능을 가진 기호들을 메타문자라고 합니다. ! + \\ & ^ \[\] \~ 등의 문자들이 메타 문자이며 이들은 각각 부여받은 기능들이 있어 이를 조합하여 특정한 패턴을 정의하는 방식으로 사용됩니다.

 이해를 돕기위해 예시와 함께 살펴보도록 하겠습니다.

### 정규표현식 사용자 함수 생성

  R Studio의 **`stringr`** 패키지 Cheat Sheet에 좋은 예시함수가 있어서 이를 차용하였습니다. 정규표현식 조건에 맞는 문자는 음영처리되어 보여집니다.

``` r
see <- function(rx) str_view_all("abc ABC 123\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde", rx)
```

### “^”

 **^**(윗꺽쇠)는 문자열의 시작지점을 의미합니다.

``` r
see('^') # 문자열의 시작지점을 매칭 
```

<div id="htmlwidget-1" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"html":"<ul>\n  <li><span class='match'><\/span>abc ABC 123\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

### “\$”

 **\$**(달러표시)는 문자열의 끝 또는 문서의 끝을 의미합니다.

``` r
see('$') # 문자열의 끝지점을 매칭 
```

<div id="htmlwidget-2" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-2">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde<span class='match'><\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

### “.”

 **.**(마침표)는 임의의 한 글자를 의미합니다.

``` r
see('.') # 임의의 글자를 매칭 
```

<div id="htmlwidget-3" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-3">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'> <\/span><span class='match'>A<\/span><span class='match'>B<\/span><span class='match'>C<\/span><span class='match'> <\/span><span class='match'>1<\/span><span class='match'>2<\/span><span class='match'>3<\/span><span class='match'>\t<\/span><span class='match'>.<\/span><span class='match'>!<\/span><span class='match'>?<\/span><span class='match'>\\<\/span><span class='match'>(<\/span><span class='match'>)<\/span><span class='match'>{<\/span><span class='match'>}<\/span>\n<span class='match'> <\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><span class='match'> <\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'> <\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span><span class='match'>d<\/span><span class='match'> <\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'> <\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'> <\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

 다른 규칙없이 마침표만을 단독으로 사용하니 임의의 모든 글자가 선택되었습니다.

### “?”

  **?**(물음표)는 앞에 지정된 문자가 없거나 하나가 있음을 의미합니다.

``` r
see('?')
```

    #> Error in stri_locate_all_regex(string, pattern, omit_no_match = TRUE, : Syntax error in regexp pattern. (U_REGEX_RULE_SYNTAX, context=`?`)

 물음표를 단독으로 사용하니 에러가 발생하였습니다. 이는 물음표 조건에 맞지 않기 때문입니다. ‘ab?c’(a와 c 사이에 b가 하나 있거나 아니면 아예 없거나) 조건으로 설정해보겠습니다.

``` r
see('ab?c') # 'a'와 'c' 사이에 'b'가 하나 있거나 혹은 아예 없는 경우 ('abc' or 'ac')
```

<div id="htmlwidget-4" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-4">{"x":{"html":"<ul>\n  <li><span class='match'>abc<\/span> ABC 123\t.!?\\(){}\n <span class='match'>abc<\/span>de aaa b<span class='match'>ac<\/span>ad .a.aa.aaa abbaab ab<span class='match'>abc<\/span>b<span class='match'>abc<\/span>dcb<span class='match'>abc<\/span>de<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

 정규표현식은 대소문자를 구분하기 때문에 소문자 ’abc’와 ’ac’가 선택된 것을 볼 수 있습니다. 정규표현식의 규칙은 매우 엄격하게 지켜지기 때문에 정규표현을 사용할 때는 정확한 규칙을 만들어서 사용할 필요가 있습니다.

### “\|”

 **\|**(또는)은 의미 그대로 또는을 의미합니다. **\|** 기호를 기준으로 좌우에 해당하는 문자들을 찾습니다.

``` r
see('a|c') # 'a' 또는 'c'를 찾아냄
```

<div id="htmlwidget-5" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-5">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span>b<span class='match'>c<\/span> ABC 123\t.!?\\(){}\n <span class='match'>a<\/span>b<span class='match'>c<\/span>de <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> b<span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span>d .<span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span>bb<span class='match'>a<\/span><span class='match'>a<\/span>b <span class='match'>a<\/span>b<span class='match'>a<\/span>b<span class='match'>c<\/span>b<span class='match'>a<\/span>b<span class='match'>c<\/span>d<span class='match'>c<\/span>b<span class='match'>a<\/span>b<span class='match'>c<\/span>de<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

### “+”

 **+**(더하기)는 앞 문자가 **1개이상**인 경우를 의미합니다. **+** 기호의 좌측 문자를 기준으로 찾습니다.

``` r
see('a+')  # 'a' 글자가 한 개 이상 존재하는 경우  
```

<div id="htmlwidget-6" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-6">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span>bc ABC 123\t.!?\\(){}\n <span class='match'>a<\/span>bcde <span class='match'>aaa<\/span> b<span class='match'>a<\/span>c<span class='match'>a<\/span>d .<span class='match'>a<\/span>.<span class='match'>aa<\/span>.<span class='match'>aaa<\/span> <span class='match'>a<\/span>bb<span class='match'>aa<\/span>b <span class='match'>a<\/span>b<span class='match'>a<\/span>bcb<span class='match'>a<\/span>bcdcb<span class='match'>a<\/span>bcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('ac+') # 'a' 다음에 'c'가 하나 이상 있음 
```

<div id="htmlwidget-7" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-7">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}\n abcde aaa b<span class='match'>ac<\/span>ad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

### "\*"

  **\***(곱하기)는 앞 문자가 **0개이상** 인 경우를 의미합니다.

``` r
see('a*') # 'a' 글자를 반환 
```

<div id="htmlwidget-8" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-8">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'><\/span>b<span class='match'><\/span>c<span class='match'><\/span> <span class='match'><\/span>A<span class='match'><\/span>B<span class='match'><\/span>C<span class='match'><\/span> <span class='match'><\/span>1<span class='match'><\/span>2<span class='match'><\/span>3<span class='match'><\/span>\t<span class='match'><\/span>.<span class='match'><\/span>!<span class='match'><\/span>?<span class='match'><\/span>\\<span class='match'><\/span>(<span class='match'><\/span>)<span class='match'><\/span>{<span class='match'><\/span>}<span class='match'><\/span>\n<span class='match'><\/span> <span class='match'>a<\/span><span class='match'><\/span>b<span class='match'><\/span>c<span class='match'><\/span>d<span class='match'><\/span>e<span class='match'><\/span> <span class='match'>aaa<\/span><span class='match'><\/span> <span class='match'><\/span>b<span class='match'>a<\/span><span class='match'><\/span>c<span class='match'>a<\/span><span class='match'><\/span>d<span class='match'><\/span> <span class='match'><\/span>.<span class='match'>a<\/span><span class='match'><\/span>.<span class='match'>aa<\/span><span class='match'><\/span>.<span class='match'>aaa<\/span><span class='match'><\/span> <span class='match'>a<\/span><span class='match'><\/span>b<span class='match'><\/span>b<span class='match'>aa<\/span><span class='match'><\/span>b<span class='match'><\/span> <span class='match'>a<\/span><span class='match'><\/span>b<span class='match'>a<\/span><span class='match'><\/span>b<span class='match'><\/span>c<span class='match'><\/span>b<span class='match'>a<\/span><span class='match'><\/span>b<span class='match'><\/span>c<span class='match'><\/span>d<span class='match'><\/span>c<span class='match'><\/span>b<span class='match'>a<\/span><span class='match'><\/span>b<span class='match'><\/span>c<span class='match'><\/span>d<span class='match'><\/span>e<span class='match'><\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('ac*')# 'a' 다음에 'c'가 없거나 하나 이상 있음 
```

<div id="htmlwidget-9" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-9">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span>bc ABC 123\t.!?\\(){}\n <span class='match'>a<\/span>bcde <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> b<span class='match'>ac<\/span><span class='match'>a<\/span>d .<span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span>bb<span class='match'>a<\/span><span class='match'>a<\/span>b <span class='match'>a<\/span>b<span class='match'>a<\/span>bcb<span class='match'>a<\/span>bcdcb<span class='match'>a<\/span>bcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

  **\* **(곱하기)는 **+**(더하기)와 유사하지만 결정적인 차이점으로 **+**(더하기)는 기호에 해당하는 문자가 하나라도 있는 문자를 반환하지만, **\***(곱하기)는 앞에 해당문자가 존재하지 않더라도 반환하는 특징이 있습니다.

### “\[ \]”

  **\[ \]**(대괄호)는 문자를 묶어서 표현할때 사용됩니다. 대괄호에 묶인 문자들은 각각 or로 취급이 됩니다.

``` r
see('[abc]')        # 'a' or 'b' or 'c'를 찾아 반환
```

<div id="htmlwidget-10" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-10">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span> ABC 123\t.!?\\(){}\n <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span>de <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span>d .<span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>b<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span>d<span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span>de<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('[abcde]c')     # 'c' 앞에 'a' or 'b' or 'c' or 'd' or 'e' 문자를 찾아 반환 ('ac', 'bc' 반환) 
```

<div id="htmlwidget-11" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-11">{"x":{"html":"<ul>\n  <li>a<span class='match'>bc<\/span> ABC 123\t.!?\\(){}\n a<span class='match'>bc<\/span>de aaa b<span class='match'>ac<\/span>ad .a.aa.aaa abbaab aba<span class='match'>bc<\/span>ba<span class='match'>bc<\/span><span class='match'>dc<\/span>ba<span class='match'>bc<\/span>de<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('[A-Z][A-Z]C')  # 'C'의 두 부분 앞 문자에서 A부터 Z글자까지 있는 문자를 찾아서 반환 ('ABC' 반환)
```

<div id="htmlwidget-12" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-12">{"x":{"html":"<ul>\n  <li>abc <span class='match'>ABC<\/span> 123\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

 또한, 특정한 문자를 제외하는 의미로서 대괄호 안에 **^**(윗 꺽쇠)를 사용합니다.

``` r
see('[^abc]') # 'a' or 'b' or 'c'를 제외한 문자를 반환
```

<div id="htmlwidget-13" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-13">{"x":{"html":"<ul>\n  <li>abc<span class='match'> <\/span><span class='match'>A<\/span><span class='match'>B<\/span><span class='match'>C<\/span><span class='match'> <\/span><span class='match'>1<\/span><span class='match'>2<\/span><span class='match'>3<\/span><span class='match'>\t<\/span><span class='match'>.<\/span><span class='match'>!<\/span><span class='match'>?<\/span><span class='match'>\\<\/span><span class='match'>(<\/span><span class='match'>)<\/span><span class='match'>{<\/span><span class='match'>}<\/span><span class='match'>\n<\/span><span class='match'> <\/span>abc<span class='match'>d<\/span><span class='match'>e<\/span><span class='match'> <\/span>aaa<span class='match'> <\/span>baca<span class='match'>d<\/span><span class='match'> <\/span><span class='match'>.<\/span>a<span class='match'>.<\/span>aa<span class='match'>.<\/span>aaa<span class='match'> <\/span>abbaab<span class='match'> <\/span>ababcbabc<span class='match'>d<\/span>cbabc<span class='match'>d<\/span><span class='match'>e<\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

### “{ }”

 **{ }**(중괄호)는 문자의 개수를 샐때 사용됩니다. 중괄호에는 세가지 용법이 있습니다.

-   {n,m} : 앞 문자가 n개 이상 m개 이하를 의미.
-   {n,} : 앞 문자가 n개 이상을 의미.
-   {n} : 앞 문자가 정확하게 n개를 의미.

``` r
see('a{1,2}') # 'a' 글자가 1개 이상 2개 이하를 표시 
```

<div id="htmlwidget-14" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-14">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span>bc ABC 123\t.!?\\(){}\n <span class='match'>a<\/span>bcde <span class='match'>aa<\/span><span class='match'>a<\/span> b<span class='match'>a<\/span>c<span class='match'>a<\/span>d .<span class='match'>a<\/span>.<span class='match'>aa<\/span>.<span class='match'>aa<\/span><span class='match'>a<\/span> <span class='match'>a<\/span>bb<span class='match'>aa<\/span>b <span class='match'>a<\/span>b<span class='match'>a<\/span>bcb<span class='match'>a<\/span>bcdcb<span class='match'>a<\/span>bcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('b{1,}')  # 'b' 글자가 1개 이상인 경우를 표시
```

<div id="htmlwidget-15" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-15">{"x":{"html":"<ul>\n  <li>a<span class='match'>b<\/span>c ABC 123\t.!?\\(){}\n a<span class='match'>b<\/span>cde aaa <span class='match'>b<\/span>acad .a.aa.aaa a<span class='match'>bb<\/span>aa<span class='match'>b<\/span> a<span class='match'>b<\/span>a<span class='match'>b<\/span>c<span class='match'>b<\/span>a<span class='match'>b<\/span>cdc<span class='match'>b<\/span>a<span class='match'>b<\/span>cde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('c{1}')   # 'c' 글자가 1개인 경우를 표시 
```

<div id="htmlwidget-16" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-16">{"x":{"html":"<ul>\n  <li>ab<span class='match'>c<\/span> ABC 123\t.!?\\(){}\n ab<span class='match'>c<\/span>de aaa ba<span class='match'>c<\/span>ad .a.aa.aaa abbaab abab<span class='match'>c<\/span>bab<span class='match'>c<\/span>d<span class='match'>c<\/span>bab<span class='match'>c<\/span>de<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

### “( )”

 **( )**(소괄호)는 문자를 묶어서 표현할때 사용합니다. 소괄호에 묶인 문자들은 각각 and로 취급됩니다.

``` r
see('(abc)')    # 'abc' 문자를 찾아 반화 
```

<div id="htmlwidget-17" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-17">{"x":{"html":"<ul>\n  <li><span class='match'>abc<\/span> ABC 123\t.!?\\(){}\n <span class='match'>abc<\/span>de aaa bacad .a.aa.aaa abbaab ab<span class='match'>abc<\/span>b<span class='match'>abc<\/span>dcb<span class='match'>abc<\/span>de<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('(abcde)c') # 'c' 앞에 'abcde' 문자를 찾아 반환 (조건에 맞는 결과 없음)
```

<div id="htmlwidget-18" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-18">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('a(bc)ba')  # 'a' 와 'ba' 사이에 'bc'가 존재하는 경우의 문자를 반환 ('abcba' 반환)
```

<div id="htmlwidget-19" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-19">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ab<span class='match'>abcba<\/span>bcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('a(b|c)ba') # 'a' 와 'ba' 사이에 'b' 또는 'c'가 존재하는 경우의 문자를 반환 ('abba' 반환 ) 
```

<div id="htmlwidget-20" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-20">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa <span class='match'>abba<\/span>ab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

 대괄호에 묶인 문자는 ‘or’ 취급이지만, 소괄호는 위의 경우처럼 별도의 or 표시 없이는 ‘and’ 취급이 되는것이 차이점입니다.

### “\\”

 앞서 **.**(마침표)는 임의의 한 글자를 찾을때 사용하는 메타문자라고 말씀을 드렸는데요. 그렇다면 만약 정규표현식을 사용하여 마침표 자체를 찾고싶을땐 어떻게 해야할까요? ’a.a’라는 문자를 찾는 상황을 가정해보겠습니다.

``` r
see('a.a') # 'a.a' 문자 반환?
```

<div id="htmlwidget-21" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-21">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}\n abcde <span class='match'>aaa<\/span> b<span class='match'>aca<\/span>d .<span class='match'>a.a<\/span><span class='match'>a.a<\/span>a<span class='match'>a a<\/span>bbaab <span class='match'>aba<\/span>bcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

 ‘a.a’라는 문자를 찾은걸 볼 수 있지만, 그것과 더불어서 ’aaa,’ ‘aca,’ ’a a’문자 역시 매칭이 되는 것을 볼 수 있습니다. 그 이유는 공교롭게도 “.”(마침표)역시 정규표현식에서 특수한 의미를 갖는 메타문자이기 때문입니다. 정규표현식 내에서 메타문자 자체를 특정하기 위해선 그 문자 앞에 메타문자 종류중 하나인 “**\\**”(역슬래시)를 붙여야합니다.

 그렇다면 이번에는 역슬래시를 활용하여 ’a.a’라는 문자를 찾아보도록 하겠습니다. 문자 가운데에 마침표가 메타문자가 아니라는것을 인지시키기 위하여 마침표 앞에 역슬래시를 붙였습니다.

``` r
see('a\.a') # 역슬래시 사용 
```

    #> Error: "'a\."로 시작하는 문자열 중에서 '\.'는 인식할 수 없는 이스케이프입니다

 역슬래시를 사용했음에도 불구하고 에러가 발생하였습니다. 여기서 한가지 더 짚고 넘어가야 할 부분은 메타문자를 찾기 위해선 역슬래시를 두 개를 붙여야 한다는 것입니다. 왜 그러하냐 하면 입력된 문자를 R parser가 처리하는 과정에서 역슬래시를 추정하는데 두 개의 역슬래시를 요구하기 때문입니다. 역슬래시를 하나만을 붙여서는 R에서 인지하지 못합니다. 역슬래시의 기능을 사용하기 위해선 **역슬래시 두 개를 붙여줘야 합니다.** 그래야 비로소 R이 두 개 붙인 역슬래시 앞의 역슬래시가 일반문자가 아님을 인지하고 해당 패턴을 찾아냅니다.

 이처럼 역슬래시가 메타문자를 메타문자로부터 탈출시키기에 *이스케이프 문자*라 부르기도 합니다.

``` r
see('a\\.a') # 역슬래시 두 개 사용 
```

<div id="htmlwidget-22" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-22">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}\n abcde aaa bacad .<span class='match'>a.a<\/span><span class='match'>a.a<\/span>aa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

 이번에는 찾고자 하던 ’a.a’단어를 정확하게 찾아낸 것을 볼 수 있습니다.

### “\\ + 일반문자”

 앞서 역슬래시( \\ )가 일반문자로 탈출시키는 *이스케이프 문자*라 설명드렸는데요. 정규표현식에서는 이러한 이스케이프 문자의 특성을 살린 특별한 기능이 몇 가지 더 있습니다. 그건 바로 일반 이스케이프 문자와 일반문자와의 조합을 통해 새로운 기능을 수행하는 것입니다. 일반문자중 일부 문자들은 이러한 특성을 지니고 있으며, 이를 정리하자면 다음과 같습니다.

<table class="table table-striped table-hover table-condensed" style="font-size: 13px; ">
<thead>
<tr>
<th style="text-align:center;">
＼ + 일반문자
</th>
<th style="text-align:center;">
의미
</th>
<th style="text-align:center;">
예시
</th>
<th style="text-align:center;">
결과
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center;min-width: 12em; ">
＼n
</td>
<td style="text-align:center;min-width: 7em; ">
줄바꿈
</td>
<td style="text-align:center;min-width: 10em; ">
see(“\\n”)
</td>
<td style="text-align:center;min-width: 60em; ">

<div id="htmlwidget-23" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-23">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}<span class='match'>\n<\/span> abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 12em; ">
＼t
</td>
<td style="text-align:center;min-width: 7em; ">
탭
</td>
<td style="text-align:center;min-width: 10em; ">
see(“\\t”)
</td>
<td style="text-align:center;min-width: 60em; ">

<div id="htmlwidget-24" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-24">{"x":{"html":"<ul>\n  <li>abc ABC 123<span class='match'>\t<\/span>.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 12em; ">
＼s
</td>
<td style="text-align:center;min-width: 7em; ">
공백
</td>
<td style="text-align:center;min-width: 10em; ">
see(“\\s”)
</td>
<td style="text-align:center;min-width: 60em; ">

<div id="htmlwidget-25" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-25">{"x":{"html":"<ul>\n  <li>abc<span class='match'> <\/span>ABC<span class='match'> <\/span>123<span class='match'>\t<\/span>.!?\\(){}<span class='match'>\n<\/span><span class='match'> <\/span>abcde<span class='match'> <\/span>aaa<span class='match'> <\/span>bacad<span class='match'> <\/span>.a.aa.aaa<span class='match'> <\/span>abbaab<span class='match'> <\/span>ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 12em; ">
＼S
</td>
<td style="text-align:center;min-width: 7em; ">
공백아님
</td>
<td style="text-align:center;min-width: 10em; ">
see(“\\S”)
</td>
<td style="text-align:center;min-width: 60em; ">

<div id="htmlwidget-26" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-26">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span> <span class='match'>A<\/span><span class='match'>B<\/span><span class='match'>C<\/span> <span class='match'>1<\/span><span class='match'>2<\/span><span class='match'>3<\/span>\t<span class='match'>.<\/span><span class='match'>!<\/span><span class='match'>?<\/span><span class='match'>\\<\/span><span class='match'>(<\/span><span class='match'>)<\/span><span class='match'>{<\/span><span class='match'>}<\/span>\n <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span> <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span><span class='match'>d<\/span> <span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>b<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 12em; ">
＼d
</td>
<td style="text-align:center;min-width: 7em; ">
숫자
</td>
<td style="text-align:center;min-width: 10em; ">
see(“\\d”)
</td>
<td style="text-align:center;min-width: 60em; ">

<div id="htmlwidget-27" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-27">{"x":{"html":"<ul>\n  <li>abc ABC <span class='match'>1<\/span><span class='match'>2<\/span><span class='match'>3<\/span>\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 12em; ">
＼D
</td>
<td style="text-align:center;min-width: 7em; ">
숫자아님
</td>
<td style="text-align:center;min-width: 10em; ">
see(“\\D”)
</td>
<td style="text-align:center;min-width: 60em; ">

<div id="htmlwidget-28" class="str_view html-widget" style="width:620px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-28">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'> <\/span><span class='match'>A<\/span><span class='match'>B<\/span><span class='match'>C<\/span><span class='match'> <\/span>123<span class='match'>\t<\/span><span class='match'>.<\/span><span class='match'>!<\/span><span class='match'>?<\/span><span class='match'>\\<\/span><span class='match'>(<\/span><span class='match'>)<\/span><span class='match'>{<\/span><span class='match'>}<\/span><span class='match'>\n<\/span><span class='match'> <\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><span class='match'> <\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'> <\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span><span class='match'>d<\/span><span class='match'> <\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'> <\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'> <\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 12em; ">
＼w
</td>
<td style="text-align:center;min-width: 7em; ">
문자
</td>
<td style="text-align:center;min-width: 10em; ">
see(“\\w”)
</td>
<td style="text-align:center;min-width: 60em; ">

<div id="htmlwidget-29" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-29">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span> <span class='match'>A<\/span><span class='match'>B<\/span><span class='match'>C<\/span> <span class='match'>1<\/span><span class='match'>2<\/span><span class='match'>3<\/span>\t.!?\\(){}\n <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span> <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span><span class='match'>d<\/span> .<span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>b<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 12em; ">
＼W
</td>
<td style="text-align:center;min-width: 7em; ">
문자아님
</td>
<td style="text-align:center;min-width: 10em; ">
see(“\\W”)
</td>
<td style="text-align:center;min-width: 60em; ">

<div id="htmlwidget-30" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-30">{"x":{"html":"<ul>\n  <li>abc<span class='match'> <\/span>ABC<span class='match'> <\/span>123<span class='match'>\t<\/span><span class='match'>.<\/span><span class='match'>!<\/span><span class='match'>?<\/span><span class='match'>\\<\/span><span class='match'>(<\/span><span class='match'>)<\/span><span class='match'>{<\/span><span class='match'>}<\/span><span class='match'>\n<\/span><span class='match'> <\/span>abcde<span class='match'> <\/span>aaa<span class='match'> <\/span>bacad<span class='match'> <\/span><span class='match'>.<\/span>a<span class='match'>.<\/span>aa<span class='match'>.<\/span>aaa<span class='match'> <\/span>abbaab<span class='match'> <\/span>ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 12em; ">
＼b
</td>
<td style="text-align:center;min-width: 7em; ">
단어경계
</td>
<td style="text-align:center;min-width: 10em; ">
see(“\\b”)
</td>
<td style="text-align:center;min-width: 60em; ">

<div id="htmlwidget-31" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-31">{"x":{"html":"<ul>\n  <li><span class='match'><\/span>abc<span class='match'><\/span> <span class='match'><\/span>ABC<span class='match'><\/span> <span class='match'><\/span>123<span class='match'><\/span>\t.!?\\(){}\n <span class='match'><\/span>abcde<span class='match'><\/span> <span class='match'><\/span>aaa<span class='match'><\/span> <span class='match'><\/span>bacad<span class='match'><\/span> .<span class='match'><\/span>a<span class='match'><\/span>.<span class='match'><\/span>aa<span class='match'><\/span>.<span class='match'><\/span>aaa<span class='match'><\/span> <span class='match'><\/span>abbaab<span class='match'><\/span> <span class='match'><\/span>ababcbabcdcbabcde<span class='match'><\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
</tbody>
</table>

### 특수표현

 이 밖에도 정규표현식에서는 **“\[: :\]”** 형태로 존재하는 특수한 표현식이 존재합니다.
<table class="table table-striped table-hover table-condensed" style="font-size: 13px; ">
<thead>
<tr>
<th style="text-align:center;">
［: :］
</th>
<th style="text-align:center;">
의미
</th>
<th style="text-align:center;">
예시
</th>
<th style="text-align:center;">
결과
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center;min-width: 5em; ">
［:digit:］
</td>
<td style="text-align:center;min-width: 10em; ">
숫자
</td>
<td style="text-align:center;min-width: 15em; ">
see(“［:digit:］”)
</td>
<td style="text-align:center;min-width: 58em; ">

<div id="htmlwidget-32" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-32">{"x":{"html":"<ul>\n  <li>abc ABC <span class='match'>1<\/span><span class='match'>2<\/span><span class='match'>3<\/span>\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 5em; ">
［:alpha:］
</td>
<td style="text-align:center;min-width: 10em; ">
문자
</td>
<td style="text-align:center;min-width: 15em; ">
see(“［:alpha:］”)
</td>
<td style="text-align:center;min-width: 58em; ">

<div id="htmlwidget-33" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-33">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span> <span class='match'>A<\/span><span class='match'>B<\/span><span class='match'>C<\/span> 123\t.!?\\(){}\n <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span> <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span><span class='match'>d<\/span> .<span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>b<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 5em; ">
［:lower:］
</td>
<td style="text-align:center;min-width: 10em; ">
소문자
</td>
<td style="text-align:center;min-width: 15em; ">
see(“［:lower:］”)
</td>
<td style="text-align:center;min-width: 58em; ">

<div id="htmlwidget-34" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-34">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span> ABC 123\t.!?\\(){}\n <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span> <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span><span class='match'>d<\/span> .<span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>b<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 5em; ">
［:upper:］
</td>
<td style="text-align:center;min-width: 10em; ">
대문자
</td>
<td style="text-align:center;min-width: 15em; ">
see(“［:upper:］”)
</td>
<td style="text-align:center;min-width: 58em; ">

<div id="htmlwidget-35" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-35">{"x":{"html":"<ul>\n  <li>abc <span class='match'>A<\/span><span class='match'>B<\/span><span class='match'>C<\/span> 123\t.!?\\(){}\n abcde aaa bacad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 5em; ">
［:alnum:］
</td>
<td style="text-align:center;min-width: 10em; ">
문자+숫자
</td>
<td style="text-align:center;min-width: 15em; ">
see(“［:alnum:］”)
</td>
<td style="text-align:center;min-width: 58em; ">

<div id="htmlwidget-36" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-36">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span> <span class='match'>A<\/span><span class='match'>B<\/span><span class='match'>C<\/span> <span class='match'>1<\/span><span class='match'>2<\/span><span class='match'>3<\/span>\t.!?\\(){}\n <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span> <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span><span class='match'>d<\/span> .<span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>b<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 5em; ">
［:punct:］
</td>
<td style="text-align:center;min-width: 10em; ">
기호
</td>
<td style="text-align:center;min-width: 15em; ">
see(“［:punct:］”)
</td>
<td style="text-align:center;min-width: 58em; ">

<div id="htmlwidget-37" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-37">{"x":{"html":"<ul>\n  <li>abc ABC 123\t<span class='match'>.<\/span><span class='match'>!<\/span><span class='match'>?<\/span><span class='match'>\\<\/span><span class='match'>(<\/span><span class='match'>)<\/span><span class='match'>{<\/span><span class='match'>}<\/span>\n abcde aaa bacad <span class='match'>.<\/span>a<span class='match'>.<\/span>aa<span class='match'>.<\/span>aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 5em; ">
［:graph:］
</td>
<td style="text-align:center;min-width: 10em; ">
문자+숫자+기호
</td>
<td style="text-align:center;min-width: 15em; ">
see(“［:graph:］”)
</td>
<td style="text-align:center;min-width: 58em; ">

<div id="htmlwidget-38" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-38">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span> <span class='match'>A<\/span><span class='match'>B<\/span><span class='match'>C<\/span> <span class='match'>1<\/span><span class='match'>2<\/span><span class='match'>3<\/span>\t<span class='match'>.<\/span><span class='match'>!<\/span><span class='match'>?<\/span><span class='match'>\\<\/span><span class='match'>(<\/span><span class='match'>)<\/span><span class='match'>{<\/span><span class='match'>}<\/span>\n <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span> <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>c<\/span><span class='match'>a<\/span><span class='match'>d<\/span> <span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>.<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>b<\/span> <span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>c<\/span><span class='match'>b<\/span><span class='match'>a<\/span><span class='match'>b<\/span><span class='match'>c<\/span><span class='match'>d<\/span><span class='match'>e<\/span><\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 5em; ">
［:space:］
</td>
<td style="text-align:center;min-width: 10em; ">
띄어쓰기
</td>
<td style="text-align:center;min-width: 15em; ">
see(“［:space:］”)
</td>
<td style="text-align:center;min-width: 58em; ">

<div id="htmlwidget-39" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-39">{"x":{"html":"<ul>\n  <li>abc<span class='match'> <\/span>ABC<span class='match'> <\/span>123<span class='match'>\t<\/span>.!?\\(){}<span class='match'>\n<\/span><span class='match'> <\/span>abcde<span class='match'> <\/span>aaa<span class='match'> <\/span>bacad<span class='match'> <\/span>.a.aa.aaa<span class='match'> <\/span>abbaab<span class='match'> <\/span>ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
<tr>
<td style="text-align:center;min-width: 5em; ">
［:blank:］
</td>
<td style="text-align:center;min-width: 10em; ">
띄어쓰기+탭
</td>
<td style="text-align:center;min-width: 15em; ">
see(“［:blank:］”)
</td>
<td style="text-align:center;min-width: 58em; ">

<div id="htmlwidget-40" class="str_view html-widget" style="width:600px;height:10px;">

</div>

<script type="application/json" data-for="htmlwidget-40">{"x":{"html":"<ul>\n  <li>abc<span class='match'> <\/span>ABC<span class='match'> <\/span>123<span class='match'>\t<\/span>.!?\\(){}\n<span class='match'> <\/span>abcde<span class='match'> <\/span>aaa<span class='match'> <\/span>bacad<span class='match'> <\/span>.a.aa.aaa<span class='match'> <\/span>abbaab<span class='match'> <\/span>ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>
</td>
</tr>
</tbody>
</table>

{{% callout note %}}
참고로 위의 표에서 사용된 일부 역슬래시와 대괄호는 출력상의 이유로 인해 일반적으로 사용하는 문자가 아닌 특수문자로 대체하였습니다. 예시함수를 사용 할 때 혹시나 에러가 발생한다면 역슬래시와 대괄호를 원래 키로 수정하여 사용하시기 바랍니다.
{{% /callout %}}

### 위치 탐색자

  다음은 문자를 기준으로 바로 앞뒤 문자를 찾거나 그 반대인 상황을 찾을때 사용하는 표현식입니다.

``` r
see('a(?=c)')  # 'a'가 'c' 바로 뒤에 있는 경우를 반환         
```

<div id="htmlwidget-41" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-41">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}\n abcde aaa b<span class='match'>a<\/span>cad .a.aa.aaa abbaab ababcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('a(?!c)')  # 'a'가 'c' 바로 뒤따르지 않는 경우를 반환 
```

<div id="htmlwidget-42" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-42">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span>bc ABC 123\t.!?\\(){}\n <span class='match'>a<\/span>bcde <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> bac<span class='match'>a<\/span>d .<span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span>bb<span class='match'>a<\/span><span class='match'>a<\/span>b <span class='match'>a<\/span>b<span class='match'>a<\/span>bcb<span class='match'>a<\/span>bcdcb<span class='match'>a<\/span>bcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('(?<=b)a') # 'a'가 'b' 바로 앞에 있는 경우를 반환 
```

<div id="htmlwidget-43" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-43">{"x":{"html":"<ul>\n  <li>abc ABC 123\t.!?\\(){}\n abcde aaa b<span class='match'>a<\/span>cad .a.aa.aaa abb<span class='match'>a<\/span>ab ab<span class='match'>a<\/span>bcb<span class='match'>a<\/span>bcdcb<span class='match'>a<\/span>bcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

``` r
see('(?<!b)a') # 'a'가 'b' 바로 앞에 있지 않는 경우를 반환 
```

<div id="htmlwidget-44" style="width:960px;height:100%;" class="str_view html-widget"></div>
<script type="application/json" data-for="htmlwidget-44">{"x":{"html":"<ul>\n  <li><span class='match'>a<\/span>bc ABC 123\t.!?\\(){}\n <span class='match'>a<\/span>bcde <span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> bac<span class='match'>a<\/span>d .<span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span>.<span class='match'>a<\/span><span class='match'>a<\/span><span class='match'>a<\/span> <span class='match'>a<\/span>bba<span class='match'>a<\/span>b <span class='match'>a<\/span>babcbabcdcbabcde<\/li>\n<\/ul>"},"evals":[],"jsHooks":[]}</script>

------------------------------------------------------------------------

 정리를 하다보니 분량이 꽤나 길어진 것 같습니다. 정규표현식은 비단 R에서 뿐만 아니라 다른 프로그래밍 언어에서도 적극 사용되는 용법이어서 한번 익혀두면 두고두고 써먹을 수 있는 장점이 있는데요. 여기에 소개하지 못한 다른 정규표현식도 많이 존재합니다. 더 자세히 공부를 하고자 하시는 분들을 위한 링크를 밑에 삽입하였으니 참고하시면 도움이 될 수 있겠습니다. R에서 텍스트데이터를 다루는 패키지 중 하나인 **`stringr`** 패키지와 정규표현식을 결합하여 사용하면 더욱 효과적으로 사용할 수 있는데요. 이는 다음장에서 다뤄보도록 하겠습니다.

<https://regexone.com/>

------------------------------------------------------------------------

# 참고자료

-   [Microsoft Docs 정규식 구문](https://docs.microsoft.com/ko-kr/previous-versions/visualstudio/visual-studio-2010/ae5bf541(v=vs.100)?redirectedfrom=MSDN)

-   [Wikipedia 정규표현식](https://ko.wikipedia.org/wiki/%EC%A0%95%EA%B7%9C_%ED%91%9C%ED%98%84%EC%8B%9D)

-   [나무위키 정규표현식](https://namu.wiki/w/%EC%A0%95%EA%B7%9C%20%ED%91%9C%ED%98%84%EC%8B%9D?from=%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D)
