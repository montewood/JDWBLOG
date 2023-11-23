---
title: "[R] tidyr 패키지로 하는 데이터 피봇(pivot_longer / pivot_wider)"
author: JDW
date: '2021-04-16'
slug: pivot_with_tidyr
categories:
  - R
tags:
  - R
  - 전처리
  - tidyr
  - 피봇
  - pivot_longer
  - pivot_wider
subtitle: ''
summary: ''
authors: []
lastmod: '2021-04-16T17:30:53+09:00'
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

<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />
<link href="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.min.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<script src="{{< blogdown/postref >}}index_files/plotly-binding/plotly.js"></script>
<script src="{{< blogdown/postref >}}index_files/typedarray/typedarray.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/jquery/jquery.min.js"></script>
<link href="{{< blogdown/postref >}}index_files/crosstalk/css/crosstalk.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/crosstalk/js/crosstalk.min.js"></script>
<link href="{{< blogdown/postref >}}index_files/plotly-htmlwidgets-css/plotly-htmlwidgets.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/plotly-main/plotly-latest.min.js"></script>
<link href="{{< blogdown/postref >}}index_files/panelset/panelset.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/panelset/panelset.js"></script>
<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />
<link href="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.min.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.js"></script>
<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />
<link href="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.min.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.js"></script>
<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />
<link href="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.min.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.js"></script>
<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />
<link href="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.min.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.js"></script>
<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />
<link href="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.min.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.js"></script>
<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />
<link href="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.min.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/bsTable/bootstrapTable.js"></script>

<img src="images/featured.png" alt="" width="40%"/>

 데이터를 원하는 모양새로 자르고 변환시키는일은 원활한 분석을 위해 꼭 필요한 작업중에 하나입니다. 오늘은 데이터 전처리 기법 중 하나인 피봇팅을 통한 long form, wide form 로의 데이터 형 변환에 대하여 알아보도록 하겠습니다.

# 피봇

 먼저 피봇이란 단어의 의미부터 짚고 넘어가겠습니다. 엑셀을 사용해보신 분이라면 피봇테이블이란 단어를 한번쯤은 접해보셨을 겁니다. 엑셀에서는 이 피봇테이블의 기능을 통해 데이터의 다양한 형 변환을 진행 할 수 있는데요. 이러한 기능적 특징을 통해 피봇의 뜻을 유추해보자면 피봇이란 단어에는 변환의 의미가 있음을 추측해 볼 수 있습니다.

<img src="images/pivot_meaning.png" alt="" width="80%">
<center>
<em> PIVOT의 사전적 의미, 출처: 네이버 영어사전 </em>
</center>

<br>

 사전에 등재되어 있는 “축을 중심으로 회전하다” 라는 동사의 의미처럼 데이터를 피봇 한다는 것은 데이터를 특정한 변수를 기준으로 위아래로 길게 늘이거나(Long form), 그 반대로 옆으로 늘이는(Wide form)형태로 만든다는 의미를 지니고 있습니다.

## Long form & Wide form

 그렇다면 이러한 데이터의 피봇팅은 어떻게 할 수 있을까요? 데이터 처리에 특화 되어있는 R에서는 여러 패키지들이 데이터 피봇팅을 할 수 있도록 하는 함수를 지원하고 있습니다. 피봇팅을 통한 long form, wide form을 변환하기 위한 데이터로서 `gapminder` 데이터를 통해 알아보도록 하겠습니다.

``` r
library(gapminder)

gapminder %>% head()
#> # A tibble: 6 x 6
#>   country     continent  year lifeExp      pop gdpPercap
#>   <fct>       <fct>     <int>   <dbl>    <int>     <dbl>
#> 1 Afghanistan Asia       1952    28.8  8425333      779.
#> 2 Afghanistan Asia       1957    30.3  9240934      821.
#> 3 Afghanistan Asia       1962    32.0 10267083      853.
#> 4 Afghanistan Asia       1967    34.0 11537966      836.
#> 5 Afghanistan Asia       1972    36.1 13079460      740.
#> 6 Afghanistan Asia       1977    38.4 14880372      786.

gapminder %>% tail()
#> # A tibble: 6 x 6
#>   country  continent  year lifeExp      pop gdpPercap
#>   <fct>    <fct>     <int>   <dbl>    <int>     <dbl>
#> 1 Zimbabwe Africa     1982    60.4  7636524      789.
#> 2 Zimbabwe Africa     1987    62.4  9216418      706.
#> 3 Zimbabwe Africa     1992    60.4 10704340      693.
#> 4 Zimbabwe Africa     1997    46.8 11404948      792.
#> 5 Zimbabwe Africa     2002    40.0 11926563      672.
#> 6 Zimbabwe Africa     2007    43.5 12311143      470.

gapminder %>% summary()
#>         country        continent        year         lifeExp     
#>  Afghanistan:  12   Africa  :624   Min.   :1952   Min.   :23.60  
#>  Albania    :  12   Americas:300   1st Qu.:1966   1st Qu.:48.20  
#>  Algeria    :  12   Asia    :396   Median :1980   Median :60.71  
#>  Angola     :  12   Europe  :360   Mean   :1980   Mean   :59.47  
#>  Argentina  :  12   Oceania : 24   3rd Qu.:1993   3rd Qu.:70.85  
#>  Australia  :  12                  Max.   :2007   Max.   :82.60  
#>  (Other)    :1632                                                
#>       pop              gdpPercap       
#>  Min.   :6.001e+04   Min.   :   241.2  
#>  1st Qu.:2.794e+06   1st Qu.:  1202.1  
#>  Median :7.024e+06   Median :  3531.8  
#>  Mean   :2.960e+07   Mean   :  7215.3  
#>  3rd Qu.:1.959e+07   3rd Qu.:  9325.5  
#>  Max.   :1.319e+09   Max.   :113523.1  
#> 

gapminder %>% glimpse()
#> Rows: 1,704
#> Columns: 6
#> $ country   <fct> "Afghanistan", "Afghanistan", "Afghanistan", "Afghanistan", ~
#> $ continent <fct> Asia, Asia, Asia, Asia, Asia, Asia, Asia, Asia, Asia, Asia, ~
#> $ year      <int> 1952, 1957, 1962, 1967, 1972, 1977, 1982, 1987, 1992, 1997, ~
#> $ lifeExp   <dbl> 28.801, 30.332, 31.997, 34.020, 36.088, 38.438, 39.854, 40.8~
#> $ pop       <int> 8425333, 9240934, 10267083, 11537966, 13079460, 14880372, 12~
#> $ gdpPercap <dbl> 779.4453, 820.8530, 853.1007, 836.1971, 739.9811, 786.1134, ~
```

 `gapminder` 데이터셋은 전 세계 142개 국가들의 gdp와 국가의 평균 수명, 인구 수의 변화 등을 기록한 일종의 시계열 데이터 입니다. 이 데이터는 기본적으로 `long form`의 형태로 제공되고 있습니다.

<table>
<thead>
<tr>
<th style="text-align:left;">
country
</th>
<th style="text-align:left;">
continent
</th>
<th style="text-align:right;">
year
</th>
<th style="text-align:right;">
lifeExp
</th>
<th style="text-align:right;">
pop
</th>
<th style="text-align:right;">
gdpPercap
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1952
</td>
<td style="text-align:right;">
28.801
</td>
<td style="text-align:right;">
8425333
</td>
<td style="text-align:right;">
779.4453
</td>
</tr>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1957
</td>
<td style="text-align:right;">
30.332
</td>
<td style="text-align:right;">
9240934
</td>
<td style="text-align:right;">
820.8530
</td>
</tr>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1962
</td>
<td style="text-align:right;">
31.997
</td>
<td style="text-align:right;">
10267083
</td>
<td style="text-align:right;">
853.1007
</td>
</tr>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1967
</td>
<td style="text-align:right;">
34.020
</td>
<td style="text-align:right;">
11537966
</td>
<td style="text-align:right;">
836.1971
</td>
</tr>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1972
</td>
<td style="text-align:right;">
36.088
</td>
<td style="text-align:right;">
13079460
</td>
<td style="text-align:right;">
739.9811
</td>
</tr>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1977
</td>
<td style="text-align:right;">
38.438
</td>
<td style="text-align:right;">
14880372
</td>
<td style="text-align:right;">
786.1134
</td>
</tr>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1982
</td>
<td style="text-align:right;">
39.854
</td>
<td style="text-align:right;">
12881816
</td>
<td style="text-align:right;">
978.0114
</td>
</tr>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1987
</td>
<td style="text-align:right;">
40.822
</td>
<td style="text-align:right;">
13867957
</td>
<td style="text-align:right;">
852.3959
</td>
</tr>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1992
</td>
<td style="text-align:right;">
41.674
</td>
<td style="text-align:right;">
16317921
</td>
<td style="text-align:right;">
649.3414
</td>
</tr>
<tr>
<td style="text-align:left;">
Afghanistan
</td>
<td style="text-align:left;">
Asia
</td>
<td style="text-align:right;">
1997
</td>
<td style="text-align:right;">
41.763
</td>
<td style="text-align:right;">
22227415
</td>
<td style="text-align:right;">
635.3414
</td>
</tr>
</tbody>
</table>

 `long form`의 형식을 살펴보자면 각각의 변수열에는 동일한 성격의 값이 있음을 알 수 있습니다. `long form` 형태의 데이터의 장점으로는 데이터를 다루기 쉽다는 점에 있습니다. 인간이 인지하기에는 다소 편리한 형태는 아니지만 컴퓨터가(특히 R에서) 연산하기에는 좋은 형태이기 때문에 시각화를 진행한다거나 머신러닝 모델을 생성할 시 유용하게 활용될 수 있습니다.

``` r
# long form 데이터는 다루기가 용이함 
library(ggplot2)
library(plotly)

g <- gapminder %>% 
  filter(year == 2007) %>% 
  ggplot(aes(x = gdpPercap, y = lifeExp, color = continent, size = pop, ids = country)) +
  geom_point(alpha = 0.5) +
  ggtitle("Life expectancy versus GDP, 2007") +
  xlab("GDP per capita (US$)") +
  ylab("Life expectancy (years)") +
  scale_color_discrete(name = "Continent") +
  scale_size_continuous(name = "Population") + 
  theme_bw()

ggplotly(g)
```

<div id="htmlwidget-1" style="width:100%;height:480px;" class="plotly html-widget"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"data":[{"x":[6223.367465,4797.231267,1441.284873,12569.85177,1217.032994,430.0706916,2042.09524,706.016537,1704.063724,986.1478792,277.5518587,3632.557798,1544.750112,2082.481567,5581.180998,12154.08975,641.3695236,690.8055759,13206.48452,752.7497265,1327.60891,942.6542111,579.231743,1463.249282,1569.331442,414.5073415,12057.49928,1044.770126,759.3499101,1042.581557,1803.151496,10956.99112,3820.17523,823.6856205,4811.060429,619.6768924,2013.977305,7670.122558,863.0884639,1598.435089,1712.472136,862.5407561,926.1410683,9269.657808,2602.394995,4513.480643,1107.482182,882.9699438,7092.923025,1056.380121,1271.211593,469.7092981],"y":[72.301,42.731,56.728,50.728,52.295,49.58,50.43,44.741,50.651,65.152,46.462,55.322,48.328,54.791,71.338,51.579,58.04,52.947,56.735,59.448,60.022,56.007,46.388,54.11,42.592,45.678,73.952,59.443,48.303,54.467,64.164,72.801,71.164,42.082,52.906,56.867,46.859,76.442,46.242,65.528,63.062,42.568,48.159,49.339,58.556,39.613,52.517,58.42,73.923,51.542,42.384,43.487],"text":["gdpPercap:  6223.3675<br />lifeExp: 72.301<br />continent: Africa<br />pop:   33333216<br />country: Algeria","gdpPercap:  4797.2313<br />lifeExp: 42.731<br />continent: Africa<br />pop:   12420476<br />country: Angola","gdpPercap:  1441.2849<br />lifeExp: 56.728<br />continent: Africa<br />pop:    8078314<br />country: Benin","gdpPercap: 12569.8518<br />lifeExp: 50.728<br />continent: Africa<br />pop:    1639131<br />country: Botswana","gdpPercap:  1217.0330<br />lifeExp: 52.295<br />continent: Africa<br />pop:   14326203<br />country: Burkina Faso","gdpPercap:   430.0707<br />lifeExp: 49.580<br />continent: Africa<br />pop:    8390505<br />country: Burundi","gdpPercap:  2042.0952<br />lifeExp: 50.430<br />continent: Africa<br />pop:   17696293<br />country: Cameroon","gdpPercap:   706.0165<br />lifeExp: 44.741<br />continent: Africa<br />pop:    4369038<br />country: Central African Republic","gdpPercap:  1704.0637<br />lifeExp: 50.651<br />continent: Africa<br />pop:   10238807<br />country: Chad","gdpPercap:   986.1479<br />lifeExp: 65.152<br />continent: Africa<br />pop:     710960<br />country: Comoros","gdpPercap:   277.5519<br />lifeExp: 46.462<br />continent: Africa<br />pop:   64606759<br />country: Congo, Dem. Rep.","gdpPercap:  3632.5578<br />lifeExp: 55.322<br />continent: Africa<br />pop:    3800610<br />country: Congo, Rep.","gdpPercap:  1544.7501<br />lifeExp: 48.328<br />continent: Africa<br />pop:   18013409<br />country: Cote d'Ivoire","gdpPercap:  2082.4816<br />lifeExp: 54.791<br />continent: Africa<br />pop:     496374<br />country: Djibouti","gdpPercap:  5581.1810<br />lifeExp: 71.338<br />continent: Africa<br />pop:   80264543<br />country: Egypt","gdpPercap: 12154.0897<br />lifeExp: 51.579<br />continent: Africa<br />pop:     551201<br />country: Equatorial Guinea","gdpPercap:   641.3695<br />lifeExp: 58.040<br />continent: Africa<br />pop:    4906585<br />country: Eritrea","gdpPercap:   690.8056<br />lifeExp: 52.947<br />continent: Africa<br />pop:   76511887<br />country: Ethiopia","gdpPercap: 13206.4845<br />lifeExp: 56.735<br />continent: Africa<br />pop:    1454867<br />country: Gabon","gdpPercap:   752.7497<br />lifeExp: 59.448<br />continent: Africa<br />pop:    1688359<br />country: Gambia","gdpPercap:  1327.6089<br />lifeExp: 60.022<br />continent: Africa<br />pop:   22873338<br />country: Ghana","gdpPercap:   942.6542<br />lifeExp: 56.007<br />continent: Africa<br />pop:    9947814<br />country: Guinea","gdpPercap:   579.2317<br />lifeExp: 46.388<br />continent: Africa<br />pop:    1472041<br />country: Guinea-Bissau","gdpPercap:  1463.2493<br />lifeExp: 54.110<br />continent: Africa<br />pop:   35610177<br />country: Kenya","gdpPercap:  1569.3314<br />lifeExp: 42.592<br />continent: Africa<br />pop:    2012649<br />country: Lesotho","gdpPercap:   414.5073<br />lifeExp: 45.678<br />continent: Africa<br />pop:    3193942<br />country: Liberia","gdpPercap: 12057.4993<br />lifeExp: 73.952<br />continent: Africa<br />pop:    6036914<br />country: Libya","gdpPercap:  1044.7701<br />lifeExp: 59.443<br />continent: Africa<br />pop:   19167654<br />country: Madagascar","gdpPercap:   759.3499<br />lifeExp: 48.303<br />continent: Africa<br />pop:   13327079<br />country: Malawi","gdpPercap:  1042.5816<br />lifeExp: 54.467<br />continent: Africa<br />pop:   12031795<br />country: Mali","gdpPercap:  1803.1515<br />lifeExp: 64.164<br />continent: Africa<br />pop:    3270065<br />country: Mauritania","gdpPercap: 10956.9911<br />lifeExp: 72.801<br />continent: Africa<br />pop:    1250882<br />country: Mauritius","gdpPercap:  3820.1752<br />lifeExp: 71.164<br />continent: Africa<br />pop:   33757175<br />country: Morocco","gdpPercap:   823.6856<br />lifeExp: 42.082<br />continent: Africa<br />pop:   19951656<br />country: Mozambique","gdpPercap:  4811.0604<br />lifeExp: 52.906<br />continent: Africa<br />pop:    2055080<br />country: Namibia","gdpPercap:   619.6769<br />lifeExp: 56.867<br />continent: Africa<br />pop:   12894865<br />country: Niger","gdpPercap:  2013.9773<br />lifeExp: 46.859<br />continent: Africa<br />pop:  135031164<br />country: Nigeria","gdpPercap:  7670.1226<br />lifeExp: 76.442<br />continent: Africa<br />pop:     798094<br />country: Reunion","gdpPercap:   863.0885<br />lifeExp: 46.242<br />continent: Africa<br />pop:    8860588<br />country: Rwanda","gdpPercap:  1598.4351<br />lifeExp: 65.528<br />continent: Africa<br />pop:     199579<br />country: Sao Tome and Principe","gdpPercap:  1712.4721<br />lifeExp: 63.062<br />continent: Africa<br />pop:   12267493<br />country: Senegal","gdpPercap:   862.5408<br />lifeExp: 42.568<br />continent: Africa<br />pop:    6144562<br />country: Sierra Leone","gdpPercap:   926.1411<br />lifeExp: 48.159<br />continent: Africa<br />pop:    9118773<br />country: Somalia","gdpPercap:  9269.6578<br />lifeExp: 49.339<br />continent: Africa<br />pop:   43997828<br />country: South Africa","gdpPercap:  2602.3950<br />lifeExp: 58.556<br />continent: Africa<br />pop:   42292929<br />country: Sudan","gdpPercap:  4513.4806<br />lifeExp: 39.613<br />continent: Africa<br />pop:    1133066<br />country: Swaziland","gdpPercap:  1107.4822<br />lifeExp: 52.517<br />continent: Africa<br />pop:   38139640<br />country: Tanzania","gdpPercap:   882.9699<br />lifeExp: 58.420<br />continent: Africa<br />pop:    5701579<br />country: Togo","gdpPercap:  7092.9230<br />lifeExp: 73.923<br />continent: Africa<br />pop:   10276158<br />country: Tunisia","gdpPercap:  1056.3801<br />lifeExp: 51.542<br />continent: Africa<br />pop:   29170398<br />country: Uganda","gdpPercap:  1271.2116<br />lifeExp: 42.384<br />continent: Africa<br />pop:   11746035<br />country: Zambia","gdpPercap:   469.7093<br />lifeExp: 43.487<br />continent: Africa<br />pop:   12311143<br />country: Zimbabwe"],"ids":["Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon","Central African Republic","Chad","Comoros","Congo, Dem. Rep.","Congo, Rep.","Cote d'Ivoire","Djibouti","Egypt","Equatorial Guinea","Eritrea","Ethiopia","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau","Kenya","Lesotho","Liberia","Libya","Madagascar","Malawi","Mali","Mauritania","Mauritius","Morocco","Mozambique","Namibia","Niger","Nigeria","Reunion","Rwanda","Sao Tome and Principe","Senegal","Sierra Leone","Somalia","South Africa","Sudan","Swaziland","Tanzania","Togo","Tunisia","Uganda","Zambia","Zimbabwe"],"type":"scatter","mode":"markers","marker":{"autocolorscale":false,"color":"rgba(248,118,109,1)","opacity":0.5,"size":[6.77527173254923,5.598900411863,5.24035330707835,4.40395768399738,5.7356199392208,5.26901439581437,5.9564773609874,4.8422259478741,5.42852631166044,4.15169850163956,7.9562662998771,4.76713328307226,5.97611663817011,4.06305719383881,8.43636871732964,4.08813627550005,4.90865391372741,8.32592567062337,4.36262596871136,4.41454467512776,6.25770233060098,5.40445197132612,4.3666011970112,6.87649613589243,4.480300710223,4.68010775450673,5.03693920169262,6.0461636385333,5.66517797748393,5.5697344023561,4.69148323166173,4.31315004312066,6.79437672410524,6.09253246174489,4.48845333089755,5.6338762963756,9.82271393327331,4.18215900239245,5.3111594952501,3.77952755905512,5.58747695964146,5.04848038390226,5.33382085391144,7.22380829377829,7.1561065468969,4.282361176813,6.98520027409899,5.00028803920838,5.43159101796171,6.580769078434,5.54798462578679,5.59074371141117],"symbol":"circle","line":{"width":1.88976377952756,"color":"rgba(248,118,109,1)"}},"hoveron":"points","name":"Africa","legendgroup":"Africa","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"x":[12779.37964,3822.137084,9065.800825,36319.23501,13171.63885,7006.580419,9645.06142,8948.102923,6025.374752,6873.262326,5728.353514,5186.050003,1201.637154,3548.330846,7320.880262,11977.57496,2749.320965,9809.185636,4172.838464,7408.905561,19328.70901,18008.50924,42951.65309,10611.46299,11415.80569],"y":[75.32,65.554,72.39,80.653,78.553,72.889,78.782,78.273,72.235,74.994,71.878,70.259,60.916,70.198,72.567,76.195,72.899,75.537,71.752,71.421,78.746,69.819,78.242,76.384,73.747],"text":["gdpPercap: 12779.3796<br />lifeExp: 75.320<br />continent: Americas<br />pop:   40301927<br />country: Argentina","gdpPercap:  3822.1371<br />lifeExp: 65.554<br />continent: Americas<br />pop:    9119152<br />country: Bolivia","gdpPercap:  9065.8008<br />lifeExp: 72.390<br />continent: Americas<br />pop:  190010647<br />country: Brazil","gdpPercap: 36319.2350<br />lifeExp: 80.653<br />continent: Americas<br />pop:   33390141<br />country: Canada","gdpPercap: 13171.6388<br />lifeExp: 78.553<br />continent: Americas<br />pop:   16284741<br />country: Chile","gdpPercap:  7006.5804<br />lifeExp: 72.889<br />continent: Americas<br />pop:   44227550<br />country: Colombia","gdpPercap:  9645.0614<br />lifeExp: 78.782<br />continent: Americas<br />pop:    4133884<br />country: Costa Rica","gdpPercap:  8948.1029<br />lifeExp: 78.273<br />continent: Americas<br />pop:   11416987<br />country: Cuba","gdpPercap:  6025.3748<br />lifeExp: 72.235<br />continent: Americas<br />pop:    9319622<br />country: Dominican Republic","gdpPercap:  6873.2623<br />lifeExp: 74.994<br />continent: Americas<br />pop:   13755680<br />country: Ecuador","gdpPercap:  5728.3535<br />lifeExp: 71.878<br />continent: Americas<br />pop:    6939688<br />country: El Salvador","gdpPercap:  5186.0500<br />lifeExp: 70.259<br />continent: Americas<br />pop:   12572928<br />country: Guatemala","gdpPercap:  1201.6372<br />lifeExp: 60.916<br />continent: Americas<br />pop:    8502814<br />country: Haiti","gdpPercap:  3548.3308<br />lifeExp: 70.198<br />continent: Americas<br />pop:    7483763<br />country: Honduras","gdpPercap:  7320.8803<br />lifeExp: 72.567<br />continent: Americas<br />pop:    2780132<br />country: Jamaica","gdpPercap: 11977.5750<br />lifeExp: 76.195<br />continent: Americas<br />pop:  108700891<br />country: Mexico","gdpPercap:  2749.3210<br />lifeExp: 72.899<br />continent: Americas<br />pop:    5675356<br />country: Nicaragua","gdpPercap:  9809.1856<br />lifeExp: 75.537<br />continent: Americas<br />pop:    3242173<br />country: Panama","gdpPercap:  4172.8385<br />lifeExp: 71.752<br />continent: Americas<br />pop:    6667147<br />country: Paraguay","gdpPercap:  7408.9056<br />lifeExp: 71.421<br />continent: Americas<br />pop:   28674757<br />country: Peru","gdpPercap: 19328.7090<br />lifeExp: 78.746<br />continent: Americas<br />pop:    3942491<br />country: Puerto Rico","gdpPercap: 18008.5092<br />lifeExp: 69.819<br />continent: Americas<br />pop:    1056608<br />country: Trinidad and Tobago","gdpPercap: 42951.6531<br />lifeExp: 78.242<br />continent: Americas<br />pop:  301139947<br />country: United States","gdpPercap: 10611.4630<br />lifeExp: 76.384<br />continent: Americas<br />pop:    3447496<br />country: Uruguay","gdpPercap: 11415.8057<br />lifeExp: 73.747<br />continent: Americas<br />pop:   26084662<br />country: Venezuela"],"ids":["Argentina","Bolivia","Brazil","Canada","Chile","Colombia","Costa Rica","Cuba","Dominican Republic","Ecuador","El Salvador","Guatemala","Haiti","Honduras","Jamaica","Mexico","Nicaragua","Panama","Paraguay","Peru","Puerto Rico","Trinidad and Tobago","United States","Uruguay","Venezuela"],"type":"scatter","mode":"markers","marker":{"autocolorscale":false,"color":"rgba(163,165,0,1)","opacity":0.5,"size":[7.07528369289545,5.33385387656466,10.9497220287062,6.77784403625228,5.86681801798997,7.23282911414924,4.81182340745332,5.52260397354913,5.35122378867242,5.69571313317988,5.1306777542305,5.61021330347734,5.27919109958551,5.18415329737334,4.6155666029278,9.20063061292984,4.99737544049698,4.68733172329479,5.10307852744492,6.55670348636785,4.78640121163658,4.26132881438327,12.807919245623,4.71746231550028,6.42738713411734],"symbol":"circle","line":{"width":1.88976377952756,"color":"rgba(163,165,0,1)"}},"hoveron":"points","name":"Americas","legendgroup":"Americas","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"x":[974.5803384,29796.04834,1391.253792,1713.778686,4959.114854,39724.97867,2452.210407,3540.651564,11605.71449,4471.061906,25523.2771,31656.06806,4519.461171,1593.06548,23348.13973,47306.98978,10461.05868,12451.6558,3095.772271,944,1091.359778,22316.19287,2605.94758,3190.481016,21654.83194,47143.17964,3970.095407,4184.548089,28718.27684,7458.396327,2441.576404,3025.349798,2280.769906],"y":[43.828,75.635,64.062,59.723,72.961,82.208,64.698,70.65,70.964,59.545,80.745,82.603,72.535,67.297,78.623,77.588,71.993,74.241,66.803,62.069,63.785,75.64,65.483,71.688,72.777,79.972,72.396,74.143,78.4,70.616,74.249,73.422,62.698],"text":["gdpPercap:   974.5803<br />lifeExp: 43.828<br />continent: Asia<br />pop:   31889923<br />country: Afghanistan","gdpPercap: 29796.0483<br />lifeExp: 75.635<br />continent: Asia<br />pop:     708573<br />country: Bahrain","gdpPercap:  1391.2538<br />lifeExp: 64.062<br />continent: Asia<br />pop:  150448339<br />country: Bangladesh","gdpPercap:  1713.7787<br />lifeExp: 59.723<br />continent: Asia<br />pop:   14131858<br />country: Cambodia","gdpPercap:  4959.1149<br />lifeExp: 72.961<br />continent: Asia<br />pop: 1318683096<br />country: China","gdpPercap: 39724.9787<br />lifeExp: 82.208<br />continent: Asia<br />pop:    6980412<br />country: Hong Kong, China","gdpPercap:  2452.2104<br />lifeExp: 64.698<br />continent: Asia<br />pop: 1110396331<br />country: India","gdpPercap:  3540.6516<br />lifeExp: 70.650<br />continent: Asia<br />pop:  223547000<br />country: Indonesia","gdpPercap: 11605.7145<br />lifeExp: 70.964<br />continent: Asia<br />pop:   69453570<br />country: Iran","gdpPercap:  4471.0619<br />lifeExp: 59.545<br />continent: Asia<br />pop:   27499638<br />country: Iraq","gdpPercap: 25523.2771<br />lifeExp: 80.745<br />continent: Asia<br />pop:    6426679<br />country: Israel","gdpPercap: 31656.0681<br />lifeExp: 82.603<br />continent: Asia<br />pop:  127467972<br />country: Japan","gdpPercap:  4519.4612<br />lifeExp: 72.535<br />continent: Asia<br />pop:    6053193<br />country: Jordan","gdpPercap:  1593.0655<br />lifeExp: 67.297<br />continent: Asia<br />pop:   23301725<br />country: Korea, Dem. Rep.","gdpPercap: 23348.1397<br />lifeExp: 78.623<br />continent: Asia<br />pop:   49044790<br />country: Korea, Rep.","gdpPercap: 47306.9898<br />lifeExp: 77.588<br />continent: Asia<br />pop:    2505559<br />country: Kuwait","gdpPercap: 10461.0587<br />lifeExp: 71.993<br />continent: Asia<br />pop:    3921278<br />country: Lebanon","gdpPercap: 12451.6558<br />lifeExp: 74.241<br />continent: Asia<br />pop:   24821286<br />country: Malaysia","gdpPercap:  3095.7723<br />lifeExp: 66.803<br />continent: Asia<br />pop:    2874127<br />country: Mongolia","gdpPercap:   944.0000<br />lifeExp: 62.069<br />continent: Asia<br />pop:   47761980<br />country: Myanmar","gdpPercap:  1091.3598<br />lifeExp: 63.785<br />continent: Asia<br />pop:   28901790<br />country: Nepal","gdpPercap: 22316.1929<br />lifeExp: 75.640<br />continent: Asia<br />pop:    3204897<br />country: Oman","gdpPercap:  2605.9476<br />lifeExp: 65.483<br />continent: Asia<br />pop:  169270617<br />country: Pakistan","gdpPercap:  3190.4810<br />lifeExp: 71.688<br />continent: Asia<br />pop:   91077287<br />country: Philippines","gdpPercap: 21654.8319<br />lifeExp: 72.777<br />continent: Asia<br />pop:   27601038<br />country: Saudi Arabia","gdpPercap: 47143.1796<br />lifeExp: 79.972<br />continent: Asia<br />pop:    4553009<br />country: Singapore","gdpPercap:  3970.0954<br />lifeExp: 72.396<br />continent: Asia<br />pop:   20378239<br />country: Sri Lanka","gdpPercap:  4184.5481<br />lifeExp: 74.143<br />continent: Asia<br />pop:   19314747<br />country: Syria","gdpPercap: 28718.2768<br />lifeExp: 78.400<br />continent: Asia<br />pop:   23174294<br />country: Taiwan","gdpPercap:  7458.3963<br />lifeExp: 70.616<br />continent: Asia<br />pop:   65068149<br />country: Thailand","gdpPercap:  2441.5764<br />lifeExp: 74.249<br />continent: Asia<br />pop:   85262356<br />country: Vietnam","gdpPercap:  3025.3498<br />lifeExp: 73.422<br />continent: Asia<br />pop:    4018332<br />country: West Bank and Gaza","gdpPercap:  2280.7699<br />lifeExp: 62.698<br />continent: Asia<br />pop:   22211743<br />country: Yemen, Rep."],"ids":["Afghanistan","Bahrain","Bangladesh","Cambodia","China","Hong Kong, China","India","Indonesia","Iran","Iraq","Israel","Japan","Jordan","Korea, Dem. Rep.","Korea, Rep.","Kuwait","Lebanon","Malaysia","Mongolia","Myanmar","Nepal","Oman","Pakistan","Philippines","Saudi Arabia","Singapore","Sri Lanka","Syria","Taiwan","Thailand","Vietnam","West Bank and Gaza","Yemen, Rep."],"type":"scatter","mode":"markers","marker":{"autocolorscale":false,"color":"rgba(0,191,125,1)","opacity":0.5,"size":[6.70929835135528,4.1508288847182,10.1588656081799,5.72211801075065,22.6771653543307,5.13475345805415,21.1203680953071,11.5573968303758,8.11057131553749,6.49879524068774,5.07824026167466,9.65077587363648,5.03869129857133,6.28100350114513,7.4168446506104,4.56983839481089,4.78354392302204,6.36196165910748,4.63065651672448,7.36876387589759,6.56775273775666,4.68175365529605,10.5466615830696,8.74086596862592,6.50384061768477,4.86541785279541,6.11737587407916,6.05493528307165,6.27409491409121,7.97119993399054,8.57951329437571,4.79655097614072,6.22127948111281],"symbol":"circle","line":{"width":1.88976377952756,"color":"rgba(0,191,125,1)"}},"hoveron":"points","name":"Asia","legendgroup":"Asia","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"x":[5937.029526,36126.4927,33692.60508,7446.298803,10680.79282,14619.22272,22833.30851,35278.41874,33207.0844,30470.0167,32170.37442,27538.41188,18008.94444,36180.78919,40675.99635,28569.7197,9253.896111,36797.93332,49357.19017,15389.92468,20509.64777,10808.47561,9786.534714,18678.31435,25768.25759,28821.0637,33859.74835,37506.41907,8458.276384,33203.26128],"y":[76.423,79.829,79.441,74.852,73.005,75.748,76.486,78.332,79.313,80.657,79.406,79.483,73.338,81.757,78.885,80.546,74.543,79.762,80.196,75.563,78.098,72.476,74.002,74.663,77.926,80.941,80.884,81.701,71.777,79.425],"text":["gdpPercap:  5937.0295<br />lifeExp: 76.423<br />continent: Europe<br />pop:    3600523<br />country: Albania","gdpPercap: 36126.4927<br />lifeExp: 79.829<br />continent: Europe<br />pop:    8199783<br />country: Austria","gdpPercap: 33692.6051<br />lifeExp: 79.441<br />continent: Europe<br />pop:   10392226<br />country: Belgium","gdpPercap:  7446.2988<br />lifeExp: 74.852<br />continent: Europe<br />pop:    4552198<br />country: Bosnia and Herzegovina","gdpPercap: 10680.7928<br />lifeExp: 73.005<br />continent: Europe<br />pop:    7322858<br />country: Bulgaria","gdpPercap: 14619.2227<br />lifeExp: 75.748<br />continent: Europe<br />pop:    4493312<br />country: Croatia","gdpPercap: 22833.3085<br />lifeExp: 76.486<br />continent: Europe<br />pop:   10228744<br />country: Czech Republic","gdpPercap: 35278.4187<br />lifeExp: 78.332<br />continent: Europe<br />pop:    5468120<br />country: Denmark","gdpPercap: 33207.0844<br />lifeExp: 79.313<br />continent: Europe<br />pop:    5238460<br />country: Finland","gdpPercap: 30470.0167<br />lifeExp: 80.657<br />continent: Europe<br />pop:   61083916<br />country: France","gdpPercap: 32170.3744<br />lifeExp: 79.406<br />continent: Europe<br />pop:   82400996<br />country: Germany","gdpPercap: 27538.4119<br />lifeExp: 79.483<br />continent: Europe<br />pop:   10706290<br />country: Greece","gdpPercap: 18008.9444<br />lifeExp: 73.338<br />continent: Europe<br />pop:    9956108<br />country: Hungary","gdpPercap: 36180.7892<br />lifeExp: 81.757<br />continent: Europe<br />pop:     301931<br />country: Iceland","gdpPercap: 40675.9964<br />lifeExp: 78.885<br />continent: Europe<br />pop:    4109086<br />country: Ireland","gdpPercap: 28569.7197<br />lifeExp: 80.546<br />continent: Europe<br />pop:   58147733<br />country: Italy","gdpPercap:  9253.8961<br />lifeExp: 74.543<br />continent: Europe<br />pop:     684736<br />country: Montenegro","gdpPercap: 36797.9333<br />lifeExp: 79.762<br />continent: Europe<br />pop:   16570613<br />country: Netherlands","gdpPercap: 49357.1902<br />lifeExp: 80.196<br />continent: Europe<br />pop:    4627926<br />country: Norway","gdpPercap: 15389.9247<br />lifeExp: 75.563<br />continent: Europe<br />pop:   38518241<br />country: Poland","gdpPercap: 20509.6478<br />lifeExp: 78.098<br />continent: Europe<br />pop:   10642836<br />country: Portugal","gdpPercap: 10808.4756<br />lifeExp: 72.476<br />continent: Europe<br />pop:   22276056<br />country: Romania","gdpPercap:  9786.5347<br />lifeExp: 74.002<br />continent: Europe<br />pop:   10150265<br />country: Serbia","gdpPercap: 18678.3144<br />lifeExp: 74.663<br />continent: Europe<br />pop:    5447502<br />country: Slovak Republic","gdpPercap: 25768.2576<br />lifeExp: 77.926<br />continent: Europe<br />pop:    2009245<br />country: Slovenia","gdpPercap: 28821.0637<br />lifeExp: 80.941<br />continent: Europe<br />pop:   40448191<br />country: Spain","gdpPercap: 33859.7484<br />lifeExp: 80.884<br />continent: Europe<br />pop:    9031088<br />country: Sweden","gdpPercap: 37506.4191<br />lifeExp: 81.701<br />continent: Europe<br />pop:    7554661<br />country: Switzerland","gdpPercap:  8458.2764<br />lifeExp: 71.777<br />continent: Europe<br />pop:   71158647<br />country: Turkey","gdpPercap: 33203.2613<br />lifeExp: 79.425<br />continent: Europe<br />pop:   60776238<br />country: United Kingdom"],"ids":["Albania","Austria","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Czech Republic","Denmark","Finland","France","Germany","Greece","Hungary","Iceland","Ireland","Italy","Montenegro","Netherlands","Norway","Poland","Portugal","Romania","Serbia","Slovak Republic","Slovenia","Spain","Sweden","Switzerland","Turkey","United Kingdom"],"type":"scatter","mode":"markers","marker":{"autocolorscale":false,"color":"rgba(0,176,246,1)","opacity":0.5,"size":[4.73930360976545,5.2515712457991,5.44107849786917,4.86531670289218,5.16855282584858,4.85794694387522,5.42769965273683,4.97410786436283,4.94778140481952,7.84043376621969,8.49809123744673,5.46648280699294,5.40514308400654,3.9460291298415,4.8085649745634,7.74130405516196,4.14203032052794,5.88528441607637,4.87472139425141,7.00115515259145,5.46138101214586,6.22484391610449,5.42123845033248,4.97176812735327,4.47964255788798,7.0812884751231,5.32616181946541,5.19097245574501,8.16356363915206,7.83015990857845],"symbol":"circle","line":{"width":1.88976377952756,"color":"rgba(0,176,246,1)"}},"hoveron":"points","name":"Europe","legendgroup":"Europe","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"x":[34435.36744,25185.00911],"y":[81.235,80.204],"text":["gdpPercap: 34435.3674<br />lifeExp: 81.235<br />continent: Oceania<br />pop:   20434176<br />country: Australia","gdpPercap: 25185.0091<br />lifeExp: 80.204<br />continent: Oceania<br />pop:    4115771<br />country: New Zealand"],"ids":["Australia","New Zealand"],"type":"scatter","mode":"markers","marker":{"autocolorscale":false,"color":"rgba(231,107,243,1)","opacity":0.5,"size":[6.12061399094626,4.80944439195868],"symbol":"circle","line":{"width":1.88976377952756,"color":"rgba(231,107,243,1)"}},"hoveron":"points","name":"Oceania","legendgroup":"Oceania","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null}],"layout":{"margin":{"t":43.7625570776256,"r":7.30593607305936,"b":40.1826484018265,"l":37.2602739726027},"plot_bgcolor":"rgba(255,255,255,1)","paper_bgcolor":"rgba(255,255,255,1)","font":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"title":{"text":"Life expectancy versus GDP, 2007","font":{"color":"rgba(0,0,0,1)","family":"","size":17.5342465753425},"x":0,"xref":"paper"},"xaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[-2176.430056865,51811.172085565],"tickmode":"array","ticktext":["0","10000","20000","30000","40000","50000"],"tickvals":[0,10000,20000,30000,40000,50000],"categoryorder":"array","categoryarray":["0","10000","20000","30000","40000","50000"],"nticks":null,"ticks":"outside","tickcolor":"rgba(51,51,51,1)","ticklen":3.65296803652968,"tickwidth":0.66417600664176,"showticklabels":true,"tickfont":{"color":"rgba(77,77,77,1)","family":"","size":11.689497716895},"tickangle":-0,"showline":false,"linecolor":null,"linewidth":0,"showgrid":true,"gridcolor":"rgba(235,235,235,1)","gridwidth":0.66417600664176,"zeroline":false,"anchor":"y","title":{"text":"GDP per capita (US$)","font":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187}},"hoverformat":".2f"},"yaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[37.4635,84.7525],"tickmode":"array","ticktext":["40","50","60","70","80"],"tickvals":[40,50,60,70,80],"categoryorder":"array","categoryarray":["40","50","60","70","80"],"nticks":null,"ticks":"outside","tickcolor":"rgba(51,51,51,1)","ticklen":3.65296803652968,"tickwidth":0.66417600664176,"showticklabels":true,"tickfont":{"color":"rgba(77,77,77,1)","family":"","size":11.689497716895},"tickangle":-0,"showline":false,"linecolor":null,"linewidth":0,"showgrid":true,"gridcolor":"rgba(235,235,235,1)","gridwidth":0.66417600664176,"zeroline":false,"anchor":"x","title":{"text":"Life expectancy (years)","font":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187}},"hoverformat":".2f"},"shapes":[{"type":"rect","fillcolor":"transparent","line":{"color":"rgba(51,51,51,1)","width":0.66417600664176,"linetype":"solid"},"yref":"paper","xref":"paper","x0":0,"x1":1,"y0":0,"y1":1}],"showlegend":true,"legend":{"bgcolor":"rgba(255,255,255,1)","bordercolor":"transparent","borderwidth":1.88976377952756,"font":{"color":"rgba(0,0,0,1)","family":"","size":11.689497716895},"y":0.93503937007874},"annotations":[{"text":"Population<br />Continent","x":1.02,"y":1,"showarrow":false,"ax":0,"ay":0,"font":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"xref":"paper","yref":"paper","textangle":-0,"xanchor":"left","yanchor":"bottom","legendTitle":true}],"hovermode":"closest","barmode":"relative"},"config":{"doubleClick":"reset","showSendToCloud":false},"source":"A","attrs":{"2a0c2201ad4":{"x":{},"y":{},"colour":{},"size":{},"ids":{},"type":"scatter"}},"cur_data":"2a0c2201ad4","visdat":{"2a0c2201ad4":["function (y) ","x"]},"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script>

<br>

 한편 `wide form` 형태의 데이터는 다루기는 어렵지만 인지적인 측면에서는 유리한 이점이 있습니다. 유럽 지역의 1인당 GDP를 연도별로 정리한 데이터를 각각 `long form` 의 형태와 `wide form` 형태로 만들어 비교해보도록 하겠습니다.

<div class="panelset">

<div class="panel">

<span class="panel-name">long form</span>

<table>
<thead>
<tr>
<th style="text-align:left;">
country
</th>
<th style="text-align:right;">
year
</th>
<th style="text-align:right;">
gdpPercap
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1952
</td>
<td style="text-align:right;">
1601.056
</td>
</tr>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1957
</td>
<td style="text-align:right;">
1942.284
</td>
</tr>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1962
</td>
<td style="text-align:right;">
2312.889
</td>
</tr>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1967
</td>
<td style="text-align:right;">
2760.197
</td>
</tr>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1972
</td>
<td style="text-align:right;">
3313.422
</td>
</tr>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1977
</td>
<td style="text-align:right;">
3533.004
</td>
</tr>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1982
</td>
<td style="text-align:right;">
3630.881
</td>
</tr>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1987
</td>
<td style="text-align:right;">
3738.933
</td>
</tr>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1992
</td>
<td style="text-align:right;">
2497.438
</td>
</tr>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1997
</td>
<td style="text-align:right;">
3193.055
</td>
</tr>
</tbody>
</table>

**…**

<table>
<caption>
Table 1: long form
</caption>
<thead>
<tr>
<th style="text-align:left;">
country
</th>
<th style="text-align:right;">
year
</th>
<th style="text-align:right;">
gdpPercap
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
1962
</td>
<td style="text-align:right;">
12477.18
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
1967
</td>
<td style="text-align:right;">
14142.85
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
1972
</td>
<td style="text-align:right;">
15895.12
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
1977
</td>
<td style="text-align:right;">
17428.75
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
1982
</td>
<td style="text-align:right;">
18232.42
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
1987
</td>
<td style="text-align:right;">
21664.79
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
1992
</td>
<td style="text-align:right;">
22705.09
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
1997
</td>
<td style="text-align:right;">
26074.53
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
2002
</td>
<td style="text-align:right;">
29479.00
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
2007
</td>
<td style="text-align:right;">
33203.26
</td>
</tr>
</tbody>
</table>

</div>

<div class="panel">

<span class="panel-name">wide form</span>

<table>
<caption>
Table 2: wide form
</caption>
<thead>
<tr>
<th style="text-align:left;">
country
</th>
<th style="text-align:right;">
1952
</th>
<th style="text-align:right;">
1957
</th>
<th style="text-align:right;">
1962
</th>
<th style="text-align:right;">
1967
</th>
<th style="text-align:right;">
1972
</th>
<th style="text-align:right;">
1977
</th>
<th style="text-align:right;">
1982
</th>
<th style="text-align:right;">
1987
</th>
<th style="text-align:right;">
1992
</th>
<th style="text-align:right;">
1997
</th>
<th style="text-align:right;">
2002
</th>
<th style="text-align:right;">
2007
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Albania
</td>
<td style="text-align:right;">
1601.0561
</td>
<td style="text-align:right;">
1942.284
</td>
<td style="text-align:right;">
2312.889
</td>
<td style="text-align:right;">
2760.197
</td>
<td style="text-align:right;">
3313.422
</td>
<td style="text-align:right;">
3533.004
</td>
<td style="text-align:right;">
3630.881
</td>
<td style="text-align:right;">
3738.933
</td>
<td style="text-align:right;">
2497.438
</td>
<td style="text-align:right;">
3193.055
</td>
<td style="text-align:right;">
4604.212
</td>
<td style="text-align:right;">
5937.030
</td>
</tr>
<tr>
<td style="text-align:left;">
Austria
</td>
<td style="text-align:right;">
6137.0765
</td>
<td style="text-align:right;">
8842.598
</td>
<td style="text-align:right;">
10750.721
</td>
<td style="text-align:right;">
12834.602
</td>
<td style="text-align:right;">
16661.626
</td>
<td style="text-align:right;">
19749.422
</td>
<td style="text-align:right;">
21597.084
</td>
<td style="text-align:right;">
23687.826
</td>
<td style="text-align:right;">
27042.019
</td>
<td style="text-align:right;">
29095.921
</td>
<td style="text-align:right;">
32417.608
</td>
<td style="text-align:right;">
36126.493
</td>
</tr>
<tr>
<td style="text-align:left;">
Belgium
</td>
<td style="text-align:right;">
8343.1051
</td>
<td style="text-align:right;">
9714.961
</td>
<td style="text-align:right;">
10991.207
</td>
<td style="text-align:right;">
13149.041
</td>
<td style="text-align:right;">
16672.144
</td>
<td style="text-align:right;">
19117.974
</td>
<td style="text-align:right;">
20979.846
</td>
<td style="text-align:right;">
22525.563
</td>
<td style="text-align:right;">
25575.571
</td>
<td style="text-align:right;">
27561.197
</td>
<td style="text-align:right;">
30485.884
</td>
<td style="text-align:right;">
33692.605
</td>
</tr>
<tr>
<td style="text-align:left;">
Bosnia and Herzegovina
</td>
<td style="text-align:right;">
973.5332
</td>
<td style="text-align:right;">
1353.989
</td>
<td style="text-align:right;">
1709.684
</td>
<td style="text-align:right;">
2172.352
</td>
<td style="text-align:right;">
2860.170
</td>
<td style="text-align:right;">
3528.481
</td>
<td style="text-align:right;">
4126.613
</td>
<td style="text-align:right;">
4314.115
</td>
<td style="text-align:right;">
2546.781
</td>
<td style="text-align:right;">
4766.356
</td>
<td style="text-align:right;">
6018.975
</td>
<td style="text-align:right;">
7446.299
</td>
</tr>
<tr>
<td style="text-align:left;">
Bulgaria
</td>
<td style="text-align:right;">
2444.2866
</td>
<td style="text-align:right;">
3008.671
</td>
<td style="text-align:right;">
4254.338
</td>
<td style="text-align:right;">
5577.003
</td>
<td style="text-align:right;">
6597.494
</td>
<td style="text-align:right;">
7612.240
</td>
<td style="text-align:right;">
8224.192
</td>
<td style="text-align:right;">
8239.855
</td>
<td style="text-align:right;">
6302.623
</td>
<td style="text-align:right;">
5970.389
</td>
<td style="text-align:right;">
7696.778
</td>
<td style="text-align:right;">
10680.793
</td>
</tr>
<tr>
<td style="text-align:left;">
Croatia
</td>
<td style="text-align:right;">
3119.2365
</td>
<td style="text-align:right;">
4338.232
</td>
<td style="text-align:right;">
5477.890
</td>
<td style="text-align:right;">
6960.298
</td>
<td style="text-align:right;">
9164.090
</td>
<td style="text-align:right;">
11305.385
</td>
<td style="text-align:right;">
13221.822
</td>
<td style="text-align:right;">
13822.584
</td>
<td style="text-align:right;">
8447.795
</td>
<td style="text-align:right;">
9875.605
</td>
<td style="text-align:right;">
11628.389
</td>
<td style="text-align:right;">
14619.223
</td>
</tr>
<tr>
<td style="text-align:left;">
Czech Republic
</td>
<td style="text-align:right;">
6876.1403
</td>
<td style="text-align:right;">
8256.344
</td>
<td style="text-align:right;">
10136.867
</td>
<td style="text-align:right;">
11399.445
</td>
<td style="text-align:right;">
13108.454
</td>
<td style="text-align:right;">
14800.161
</td>
<td style="text-align:right;">
15377.229
</td>
<td style="text-align:right;">
16310.443
</td>
<td style="text-align:right;">
14297.021
</td>
<td style="text-align:right;">
16048.514
</td>
<td style="text-align:right;">
17596.210
</td>
<td style="text-align:right;">
22833.309
</td>
</tr>
<tr>
<td style="text-align:left;">
Denmark
</td>
<td style="text-align:right;">
9692.3852
</td>
<td style="text-align:right;">
11099.659
</td>
<td style="text-align:right;">
13583.314
</td>
<td style="text-align:right;">
15937.211
</td>
<td style="text-align:right;">
18866.207
</td>
<td style="text-align:right;">
20422.901
</td>
<td style="text-align:right;">
21688.040
</td>
<td style="text-align:right;">
25116.176
</td>
<td style="text-align:right;">
26406.740
</td>
<td style="text-align:right;">
29804.346
</td>
<td style="text-align:right;">
32166.500
</td>
<td style="text-align:right;">
35278.419
</td>
</tr>
<tr>
<td style="text-align:left;">
Finland
</td>
<td style="text-align:right;">
6424.5191
</td>
<td style="text-align:right;">
7545.415
</td>
<td style="text-align:right;">
9371.843
</td>
<td style="text-align:right;">
10921.636
</td>
<td style="text-align:right;">
14358.876
</td>
<td style="text-align:right;">
15605.423
</td>
<td style="text-align:right;">
18533.158
</td>
<td style="text-align:right;">
21141.012
</td>
<td style="text-align:right;">
20647.165
</td>
<td style="text-align:right;">
23723.950
</td>
<td style="text-align:right;">
28204.591
</td>
<td style="text-align:right;">
33207.084
</td>
</tr>
<tr>
<td style="text-align:left;">
France
</td>
<td style="text-align:right;">
7029.8093
</td>
<td style="text-align:right;">
8662.835
</td>
<td style="text-align:right;">
10560.486
</td>
<td style="text-align:right;">
12999.918
</td>
<td style="text-align:right;">
16107.192
</td>
<td style="text-align:right;">
18292.635
</td>
<td style="text-align:right;">
20293.897
</td>
<td style="text-align:right;">
22066.442
</td>
<td style="text-align:right;">
24703.796
</td>
<td style="text-align:right;">
25889.785
</td>
<td style="text-align:right;">
28926.032
</td>
<td style="text-align:right;">
30470.017
</td>
</tr>
<tr>
<td style="text-align:left;">
Germany
</td>
<td style="text-align:right;">
7144.1144
</td>
<td style="text-align:right;">
10187.827
</td>
<td style="text-align:right;">
12902.463
</td>
<td style="text-align:right;">
14745.626
</td>
<td style="text-align:right;">
18016.180
</td>
<td style="text-align:right;">
20512.921
</td>
<td style="text-align:right;">
22031.533
</td>
<td style="text-align:right;">
24639.186
</td>
<td style="text-align:right;">
26505.303
</td>
<td style="text-align:right;">
27788.884
</td>
<td style="text-align:right;">
30035.802
</td>
<td style="text-align:right;">
32170.374
</td>
</tr>
<tr>
<td style="text-align:left;">
Greece
</td>
<td style="text-align:right;">
3530.6901
</td>
<td style="text-align:right;">
4916.300
</td>
<td style="text-align:right;">
6017.191
</td>
<td style="text-align:right;">
8513.097
</td>
<td style="text-align:right;">
12724.830
</td>
<td style="text-align:right;">
14195.524
</td>
<td style="text-align:right;">
15268.421
</td>
<td style="text-align:right;">
16120.528
</td>
<td style="text-align:right;">
17541.496
</td>
<td style="text-align:right;">
18747.698
</td>
<td style="text-align:right;">
22514.255
</td>
<td style="text-align:right;">
27538.412
</td>
</tr>
<tr>
<td style="text-align:left;">
Hungary
</td>
<td style="text-align:right;">
5263.6738
</td>
<td style="text-align:right;">
6040.180
</td>
<td style="text-align:right;">
7550.360
</td>
<td style="text-align:right;">
9326.645
</td>
<td style="text-align:right;">
10168.656
</td>
<td style="text-align:right;">
11674.837
</td>
<td style="text-align:right;">
12545.991
</td>
<td style="text-align:right;">
12986.480
</td>
<td style="text-align:right;">
10535.629
</td>
<td style="text-align:right;">
11712.777
</td>
<td style="text-align:right;">
14843.936
</td>
<td style="text-align:right;">
18008.944
</td>
</tr>
<tr>
<td style="text-align:left;">
Iceland
</td>
<td style="text-align:right;">
7267.6884
</td>
<td style="text-align:right;">
9244.001
</td>
<td style="text-align:right;">
10350.159
</td>
<td style="text-align:right;">
13319.896
</td>
<td style="text-align:right;">
15798.064
</td>
<td style="text-align:right;">
19654.962
</td>
<td style="text-align:right;">
23269.607
</td>
<td style="text-align:right;">
26923.206
</td>
<td style="text-align:right;">
25144.392
</td>
<td style="text-align:right;">
28061.100
</td>
<td style="text-align:right;">
31163.202
</td>
<td style="text-align:right;">
36180.789
</td>
</tr>
<tr>
<td style="text-align:left;">
Ireland
</td>
<td style="text-align:right;">
5210.2803
</td>
<td style="text-align:right;">
5599.078
</td>
<td style="text-align:right;">
6631.597
</td>
<td style="text-align:right;">
7655.569
</td>
<td style="text-align:right;">
9530.773
</td>
<td style="text-align:right;">
11150.981
</td>
<td style="text-align:right;">
12618.321
</td>
<td style="text-align:right;">
13872.867
</td>
<td style="text-align:right;">
17558.816
</td>
<td style="text-align:right;">
24521.947
</td>
<td style="text-align:right;">
34077.049
</td>
<td style="text-align:right;">
40675.996
</td>
</tr>
<tr>
<td style="text-align:left;">
Italy
</td>
<td style="text-align:right;">
4931.4042
</td>
<td style="text-align:right;">
6248.656
</td>
<td style="text-align:right;">
8243.582
</td>
<td style="text-align:right;">
10022.401
</td>
<td style="text-align:right;">
12269.274
</td>
<td style="text-align:right;">
14255.985
</td>
<td style="text-align:right;">
16537.483
</td>
<td style="text-align:right;">
19207.235
</td>
<td style="text-align:right;">
22013.645
</td>
<td style="text-align:right;">
24675.024
</td>
<td style="text-align:right;">
27968.098
</td>
<td style="text-align:right;">
28569.720
</td>
</tr>
<tr>
<td style="text-align:left;">
Montenegro
</td>
<td style="text-align:right;">
2647.5856
</td>
<td style="text-align:right;">
3682.260
</td>
<td style="text-align:right;">
4649.594
</td>
<td style="text-align:right;">
5907.851
</td>
<td style="text-align:right;">
7778.414
</td>
<td style="text-align:right;">
9595.930
</td>
<td style="text-align:right;">
11222.588
</td>
<td style="text-align:right;">
11732.510
</td>
<td style="text-align:right;">
7003.339
</td>
<td style="text-align:right;">
6465.613
</td>
<td style="text-align:right;">
6557.194
</td>
<td style="text-align:right;">
9253.896
</td>
</tr>
<tr>
<td style="text-align:left;">
Netherlands
</td>
<td style="text-align:right;">
8941.5719
</td>
<td style="text-align:right;">
11276.193
</td>
<td style="text-align:right;">
12790.850
</td>
<td style="text-align:right;">
15363.251
</td>
<td style="text-align:right;">
18794.746
</td>
<td style="text-align:right;">
21209.059
</td>
<td style="text-align:right;">
21399.460
</td>
<td style="text-align:right;">
23651.324
</td>
<td style="text-align:right;">
26790.950
</td>
<td style="text-align:right;">
30246.131
</td>
<td style="text-align:right;">
33724.758
</td>
<td style="text-align:right;">
36797.933
</td>
</tr>
<tr>
<td style="text-align:left;">
Norway
</td>
<td style="text-align:right;">
10095.4217
</td>
<td style="text-align:right;">
11653.973
</td>
<td style="text-align:right;">
13450.402
</td>
<td style="text-align:right;">
16361.876
</td>
<td style="text-align:right;">
18965.056
</td>
<td style="text-align:right;">
23311.349
</td>
<td style="text-align:right;">
26298.635
</td>
<td style="text-align:right;">
31540.975
</td>
<td style="text-align:right;">
33965.661
</td>
<td style="text-align:right;">
41283.164
</td>
<td style="text-align:right;">
44683.975
</td>
<td style="text-align:right;">
49357.190
</td>
</tr>
<tr>
<td style="text-align:left;">
Poland
</td>
<td style="text-align:right;">
4029.3297
</td>
<td style="text-align:right;">
4734.253
</td>
<td style="text-align:right;">
5338.752
</td>
<td style="text-align:right;">
6557.153
</td>
<td style="text-align:right;">
8006.507
</td>
<td style="text-align:right;">
9508.141
</td>
<td style="text-align:right;">
8451.531
</td>
<td style="text-align:right;">
9082.351
</td>
<td style="text-align:right;">
7738.881
</td>
<td style="text-align:right;">
10159.584
</td>
<td style="text-align:right;">
12002.239
</td>
<td style="text-align:right;">
15389.925
</td>
</tr>
<tr>
<td style="text-align:left;">
Portugal
</td>
<td style="text-align:right;">
3068.3199
</td>
<td style="text-align:right;">
3774.572
</td>
<td style="text-align:right;">
4727.955
</td>
<td style="text-align:right;">
6361.518
</td>
<td style="text-align:right;">
9022.247
</td>
<td style="text-align:right;">
10172.486
</td>
<td style="text-align:right;">
11753.843
</td>
<td style="text-align:right;">
13039.309
</td>
<td style="text-align:right;">
16207.267
</td>
<td style="text-align:right;">
17641.032
</td>
<td style="text-align:right;">
19970.908
</td>
<td style="text-align:right;">
20509.648
</td>
</tr>
<tr>
<td style="text-align:left;">
Romania
</td>
<td style="text-align:right;">
3144.6132
</td>
<td style="text-align:right;">
3943.370
</td>
<td style="text-align:right;">
4734.998
</td>
<td style="text-align:right;">
6470.867
</td>
<td style="text-align:right;">
8011.414
</td>
<td style="text-align:right;">
9356.397
</td>
<td style="text-align:right;">
9605.314
</td>
<td style="text-align:right;">
9696.273
</td>
<td style="text-align:right;">
6598.410
</td>
<td style="text-align:right;">
7346.548
</td>
<td style="text-align:right;">
7885.360
</td>
<td style="text-align:right;">
10808.476
</td>
</tr>
<tr>
<td style="text-align:left;">
Serbia
</td>
<td style="text-align:right;">
3581.4594
</td>
<td style="text-align:right;">
4981.091
</td>
<td style="text-align:right;">
6289.629
</td>
<td style="text-align:right;">
7991.707
</td>
<td style="text-align:right;">
10522.067
</td>
<td style="text-align:right;">
12980.670
</td>
<td style="text-align:right;">
15181.093
</td>
<td style="text-align:right;">
15870.879
</td>
<td style="text-align:right;">
9325.068
</td>
<td style="text-align:right;">
7914.320
</td>
<td style="text-align:right;">
7236.075
</td>
<td style="text-align:right;">
9786.535
</td>
</tr>
<tr>
<td style="text-align:left;">
Slovak Republic
</td>
<td style="text-align:right;">
5074.6591
</td>
<td style="text-align:right;">
6093.263
</td>
<td style="text-align:right;">
7481.108
</td>
<td style="text-align:right;">
8412.902
</td>
<td style="text-align:right;">
9674.168
</td>
<td style="text-align:right;">
10922.664
</td>
<td style="text-align:right;">
11348.546
</td>
<td style="text-align:right;">
12037.268
</td>
<td style="text-align:right;">
9498.468
</td>
<td style="text-align:right;">
12126.231
</td>
<td style="text-align:right;">
13638.778
</td>
<td style="text-align:right;">
18678.314
</td>
</tr>
<tr>
<td style="text-align:left;">
Slovenia
</td>
<td style="text-align:right;">
4215.0417
</td>
<td style="text-align:right;">
5862.277
</td>
<td style="text-align:right;">
7402.303
</td>
<td style="text-align:right;">
9405.489
</td>
<td style="text-align:right;">
12383.486
</td>
<td style="text-align:right;">
15277.030
</td>
<td style="text-align:right;">
17866.722
</td>
<td style="text-align:right;">
18678.535
</td>
<td style="text-align:right;">
14214.717
</td>
<td style="text-align:right;">
17161.107
</td>
<td style="text-align:right;">
20660.019
</td>
<td style="text-align:right;">
25768.258
</td>
</tr>
<tr>
<td style="text-align:left;">
Spain
</td>
<td style="text-align:right;">
3834.0347
</td>
<td style="text-align:right;">
4564.802
</td>
<td style="text-align:right;">
5693.844
</td>
<td style="text-align:right;">
7993.512
</td>
<td style="text-align:right;">
10638.751
</td>
<td style="text-align:right;">
13236.921
</td>
<td style="text-align:right;">
13926.170
</td>
<td style="text-align:right;">
15764.983
</td>
<td style="text-align:right;">
18603.065
</td>
<td style="text-align:right;">
20445.299
</td>
<td style="text-align:right;">
24835.472
</td>
<td style="text-align:right;">
28821.064
</td>
</tr>
<tr>
<td style="text-align:left;">
Sweden
</td>
<td style="text-align:right;">
8527.8447
</td>
<td style="text-align:right;">
9911.878
</td>
<td style="text-align:right;">
12329.442
</td>
<td style="text-align:right;">
15258.297
</td>
<td style="text-align:right;">
17832.025
</td>
<td style="text-align:right;">
18855.725
</td>
<td style="text-align:right;">
20667.381
</td>
<td style="text-align:right;">
23586.929
</td>
<td style="text-align:right;">
23880.017
</td>
<td style="text-align:right;">
25266.595
</td>
<td style="text-align:right;">
29341.631
</td>
<td style="text-align:right;">
33859.748
</td>
</tr>
<tr>
<td style="text-align:left;">
Switzerland
</td>
<td style="text-align:right;">
14734.2327
</td>
<td style="text-align:right;">
17909.490
</td>
<td style="text-align:right;">
20431.093
</td>
<td style="text-align:right;">
22966.144
</td>
<td style="text-align:right;">
27195.113
</td>
<td style="text-align:right;">
26982.291
</td>
<td style="text-align:right;">
28397.715
</td>
<td style="text-align:right;">
30281.705
</td>
<td style="text-align:right;">
31871.530
</td>
<td style="text-align:right;">
32135.323
</td>
<td style="text-align:right;">
34480.958
</td>
<td style="text-align:right;">
37506.419
</td>
</tr>
<tr>
<td style="text-align:left;">
Turkey
</td>
<td style="text-align:right;">
1969.1010
</td>
<td style="text-align:right;">
2218.754
</td>
<td style="text-align:right;">
2322.870
</td>
<td style="text-align:right;">
2826.356
</td>
<td style="text-align:right;">
3450.696
</td>
<td style="text-align:right;">
4269.122
</td>
<td style="text-align:right;">
4241.356
</td>
<td style="text-align:right;">
5089.044
</td>
<td style="text-align:right;">
5678.348
</td>
<td style="text-align:right;">
6601.430
</td>
<td style="text-align:right;">
6508.086
</td>
<td style="text-align:right;">
8458.276
</td>
</tr>
<tr>
<td style="text-align:left;">
United Kingdom
</td>
<td style="text-align:right;">
9979.5085
</td>
<td style="text-align:right;">
11283.178
</td>
<td style="text-align:right;">
12477.177
</td>
<td style="text-align:right;">
14142.851
</td>
<td style="text-align:right;">
15895.116
</td>
<td style="text-align:right;">
17428.748
</td>
<td style="text-align:right;">
18232.425
</td>
<td style="text-align:right;">
21664.788
</td>
<td style="text-align:right;">
22705.093
</td>
<td style="text-align:right;">
26074.531
</td>
<td style="text-align:right;">
29478.999
</td>
<td style="text-align:right;">
33203.261
</td>
</tr>
</tbody>
</table>

</div>

</div>

 차이점이 눈에 보이시나요? `long form`의 형태는 연도마다 국가와 1인당 GDP값이 일일이 표기가 되어있어 데이터가 마치 치즈처럼 아래로 쭉 늘어난 것을 볼 수 있습니다. 출력 과정에선 그로인한 정보적 손실이 발생하고 있는 상황이죠. 이와는 대조되게 `wide form`의 데이터는 개별 연도를 각각의 column으로 배치하여 해당 국가들의 연도별 1인당GDP의 변화량을 좀 더 잘 캐치할 수 있습니다. 만약 요약된 정보를 report의 형태로 제작해야 할 상황이라면 `wide form`의 형태가 가독성 측면에서 더욱 우수하겠죠?

# 피봇팅

## wide form -&gt; long form

 이번에는 직접 R에서 피봇팅을 지원하는 패키지들 함수를 통한 데이터 형 변환을 실시 해보겠습니다. R의 기본함수 만으로도 충분히 피봇팅을 진행 할 수 있겠지만 절차도 복잡할 뿐더러 무엇보다도 이미 피봇팅을 지원하고 있는 함수들이 있기 때문에 기본함수를 사용하는 것은 비 효율적입니다. 다음은 `long form`을 지원하는 함수입니다.

     - reshape2::melt()
     - tidyr::gather() 
     - tidyr::pivot_longer()

<br>

<div class="panelset">

<div class="panel">

<span class="panel-name">melt()</span>

### reshape2::melt()

  **`reshape2`** 패키지는 이름에도 나와있듯이 데이터를 재구성하는데 유용한 함수들을 제공하는 패키지입니다. 하지만 만들어진지 5년이 넘은 패키지이기 때문에 구형의 함수이고, 패키지 제작자인 Hadley Wickham도 다른 최신의 패키지 함수를 사용하길 권하고 있는 상황입니다. `melt`함수는 일반적으로 넓게 퍼져있는 `wide form` 형태의 데이터를 아이스크림을 녹이듯 밑으로 흘러내린 `long form`의 형태로 변환시켜주는 기능을 지니고 있는데 자주 사용되는 S3 class 데이터프레임에서 사용하는 `melt` 함수의 사용법으로는 다음과 같습니다.

``` r
reshape2::melt(data, id.vars, measure.vars, variable.name = "variable", na.rm = !preserve.na, preserve.na = TRUE, ...)
```

<details>
<summary>
매개변수 설명 보기
</summary>
<table class="table table-striped table-hover table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
매개변수명
</th>
<th style="text-align:left;">
의미
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;border-right:1px solid;">
data
</td>
<td style="text-align:left;">
데이터 프레임
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
id.vars
</td>
<td style="text-align:left;">
기준 변수 열
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
measure.vars
</td>
<td style="text-align:left;">
측정 변수 열
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
variable.name
</td>
<td style="text-align:left;">
기준 변수 열 이름
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
value.name
</td>
<td style="text-align:left;">
측정 변수 열 이름
</td>
</tr>
</tbody>
</table>
</details>

 `melt()`함수를 사용하여 `wide form`의 데이터를 `long form`으로 변환하는 과정을 진행해 보도록 하죠. 미리 위에서 알아봤던 연도별 유럽 지역의 1인당 GDP 데이터를 `gapminder_wide`변수에 저장했고, 이를 사용해 보도록 하겠습니다.

``` r
# long form 데이터 확인 
gapminder_long <- gapminder %>% 
  filter(continent == 'Europe') %>% 
  select(country, year, gdpPercap) %>% 
  as.data.frame()

gapminder_long %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422

# wide form 데이터 확인 
gapminder_wide %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>

# wide form -> long form 변환 
gapminder_long_melt <- gapminder_wide %>% 
  reshape2::melt(id.vars = "country", variable.name = "year", value.name = "gdpPercap") %>% 
  arrange(country)

# 결과 비교 
gapminder_long %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422

gapminder_long_melt %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422
```

 `melt()`함수의 `id.vars`에 국가들의 이름을 의미하는 `country`를 지정하고, 변수열의 이름으로 “year,” 값 열 이름엔 “gdpPercap”을 지정하였습니다. `id.vars`를 지정하면 함수는 기존에 column으로 존재하던 나머지 열 들을 하나의 변수 안에 범주형 값으로 통합시켜(`measure.vars`) 데이터를 `long form`의 형식으로 변환하는 작업을 수행합니다.

<br>
<button type="button" onclick="location.href = '#wide-form--gt-long-form' ">다른 함수 보기</button>

</div>

<div class="panel">

<span class="panel-name">gather()</span>

### tidyr::gather()

 **`tidyr`** 패키지는 **`tidyverse`** 군에 속한 패키지로서 `tidy`한 데이터를 생성하도록 돕는 함수를 제공하는 패키지 입니다. 여기서 언급한 `tidy`한 데이터란 이렇게 정의가 됩니다.

1.  Every column is variable.
2.  Every row is an observation.
3.  Every cell is a single value.

 여기서 잠깐 퀴즈 하나 내보겠습니다. 앞서 살펴보았던 `wide form`의 데이터는 `tidy`한 데이터일까요 아닐까요? 이름에서 추측하자면 정돈된 데이터라는 관점에서 생각해본다면 `tidy`한 데이터가 맞을수도 있다고 생각할 수 있습니다. 하지만 정답은 X 입니다. 데이터가 `tidy`하다 라는 것을 이해하려면 데이터의 *변수* 과 데이터의 *관측점* 를 이해하고 있어야 합니다. 위의 정의를 좀 더 풀어서 알아보도록 하죠.

-   Every column is variable. &gt;&gt; 모든 열은 *변수* 여야 한다는 말의 의미는 열에는 속성(키, 온도, 무게 등)값이 와야 합니다.

-   Every row is an observation. &gt;&gt; 각 행은 *관측점* 이여야 한다는 말의 의미는 속성(변수)마다 동일한 단위로 측정된 값이 등장해야 한다는 의미입니다.(범주형 : 사람, 국가, 연도 등, 연속형 : 숫자 값)

 지금까지 알아보았던 `wide form`형태의 데이터는 연도(관측점)가 각각 독립된 열에 존재하여 다루기가 다소 난해한 경우의 데이터 였습니다. 이처럼 데이터가 `tidy`하다 함은 일반적으로 `long form` 형태의 데이터를 지칭합니다.

  `gather`함수의 사용법으로는 다음과 같습니다.

``` r
tidyr::gather(
  data,
  key = "key",
  value = "value",
  ...,
  na.rm = FALSE,
  convert = FALSE,
  factor_key = FALSE
)
```

<details>
<summary>
매개변수 설명 보기
</summary>
<table class="table table-striped table-hover table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
매개변수명
</th>
<th style="text-align:left;">
의미
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;border-right:1px solid;">
data
</td>
<td style="text-align:left;">
데이터 프레임
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
key
</td>
<td style="text-align:left;">
key열 이름
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
value
</td>
<td style="text-align:left;">
value 열 이름
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
…
</td>
<td style="text-align:left;">
열 선택
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
convert
</td>
<td style="text-align:left;">
숫자형 변환, TRUE/FALSE
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
factor\_key
</td>
<td style="text-align:left;">
범주형 변환, TRUE/FALSE
</td>
</tr>
</tbody>
</table>
</details>

  이번에는 `gather`함수를 사용하여 이전 `melt`함수를 사용한 과정과 동일한 작업을 진행해 보도록 하겠습니다.

``` r
# long form 데이터 확인 
gapminder_long <- gapminder %>% 
  filter(continent == 'Europe') %>% 
  select(country, year, gdpPercap) %>% 
  as.data.frame()

gapminder_long %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422

# wide form 데이터 확인 
gapminder_wide %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>

# wide form -> long form 변환 
gapminder_long_gather <- gapminder_wide %>% 
  tidyr::gather(key = "year", value = "gdpPercap", -country, convert = T) %>% 
  arrange(country)

# 결과 비교 
gapminder_long %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422

gapminder_long_gather %>% head(5)
#> # A tibble: 5 x 3
#>   country  year gdpPercap
#>   <fct>   <int>     <dbl>
#> 1 Albania  1952     1601.
#> 2 Albania  1957     1942.
#> 3 Albania  1962     2313.
#> 4 Albania  1967     2760.
#> 5 Albania  1972     3313.
```

  이전의 `melt` 함수와 비교해보면 변수열 통합을 위한 기준열을 지정하는 방식이 약간 차이가 나는 것을 알 수 있습니다. `melt`함수에는 매개변수로 `id.vars`를 통해 기준열을 지정할 수 있는데, `gather`함수에서는 마이너스 기호( - )를 통해 기준열을 지정할 수 있었습니다. 마이너스 기호로 `country`를 지정하면 이후 자동으로 나머지 열을 통합하는 작업을 수행하는 점에 있어서는 `melt`함수와 별반 차이가 없고, 결과를 비교해 봐도 일치하는 점을 알 수 있습니다.

<br>
<button type="button" onclick="location.href = '#wide-form--gt-long-form' ">다른 함수 보기</button>

</div>

<div class="panel">

<span class="panel-name">pivot\_loger()</span>

### tidyr::pivot\_longer()

 **`tidyr`** 패키지에서 지원하는 `long form` 데이터 생성을 목적으로 한 두 번째 함수인 `pivot_longer`함수는 기존의 `gather`함수가 지니고 있던 직관적이지 못해 이해하기 어렵던 명칭과 기능을 개선한 함수로서 **`reshape2`** 및 **`tidyr`** 패키지의 제작자이자 R 구루이신 Hadley Wickham이 추천하는 방식입니다.

 `pivot_longer`함수의 사용법으로는 다음과 같습니다.

``` r
tidyr::pivot_longer(
  data,
  cols,
  names_to = "name",
  names_prefix = NULL,
  names_sep = NULL,
  names_pattern = NULL,
  names_ptypes = list(),
  names_transform = list(),
  names_repair = "check_unique",
  values_to = "value",
  values_drop_na = FALSE,
  values_ptypes = list(),
  values_transform = list(),
  ...
)
```

<details>
<summary>
매개변수 설명 보기
</summary>
<table class="table table-striped table-hover table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
매개변수명
</th>
<th style="text-align:left;">
의미
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;border-right:1px solid;">
data
</td>
<td style="text-align:left;">
데이터 프레임
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
col
</td>
<td style="text-align:left;">
long form으로 변환할 변수 열
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_to
</td>
<td style="text-align:left;">
새 변수열 이름
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_prefix
</td>
<td style="text-align:left;">
기존 변수열 이름에서 제거할 문자열 패턴(정규식)
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_sep
</td>
<td style="text-align:left;">
기존 변수열 이름에서 분할한 지점(숫자/문자)
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_pattern
</td>
<td style="text-align:left;">
기존 변수열 이름에서 일치하는 지점(정규식)
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_ptypes
</td>
<td style="text-align:left;">
새 변수열 이름 타입
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_transform
</td>
<td style="text-align:left;">
새 변수열 이름 타입 변환
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_repair
</td>
<td style="text-align:left;">
유효하지 않은 변수열 처리 방법 지정
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
values\_to
</td>
<td style="text-align:left;">
새 값 열 이름
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
values\_drop\_na
</td>
<td style="text-align:left;">
NA가 포함된 행 삭제(TRUE/FALSE)
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
values\_ptypes
</td>
<td style="text-align:left;">
새 값 열 이름 타입
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
values\_transform
</td>
<td style="text-align:left;">
새 값 열 이름 타입 변환
</td>
</tr>
</tbody>
</table>
</details>

 `pivot_loger`함수를 사용하여 `long form`변환을 진행해 보도록 하겠습니다.

``` r
# long form 데이터 확인 
gapminder_long <- gapminder %>% 
  filter(continent == 'Europe') %>% 
  select(country, year, gdpPercap) %>% 
  as.data.frame()

gapminder_long %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422

# wide form 데이터 확인 
gapminder_wide %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>

# wide form -> long form 변환 
gapminder_long_pivot_longer <- gapminder_wide %>% 
  tidyr::pivot_longer(col = -country, names_to = "year", values_to = "gdpPercap") %>% 
  arrange(country)

# 결과 비교 
gapminder_long %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422

gapminder_long_pivot_longer %>% head(5)
#> # A tibble: 5 x 3
#>   country year  gdpPercap
#>   <fct>   <chr>     <dbl>
#> 1 Albania 1952      1601.
#> 2 Albania 1957      1942.
#> 3 Albania 1962      2313.
#> 4 Albania 1967      2760.
#> 5 Albania 1972      3313.
```

  `gather`함수와 마찬가지로 기준이 되는 열을 지정하기 위해 마이너스 기호( - )를 사용합니다. 하지만 `gather`함수에서는 기준열의 지정을 하는 인수가 없어 실제 함수를 사용하는 상황에 있어서 이해가 필요한 시간이 필요한 반면, `pivot_longer`함수에서는 인수 `col`을 지정하여 기준열 지정을 좀 더 명확히 하는 장점이 있습니다. 또한 여기 예제에서는 다루지 않았지만 변수열과 이름열을 설정하는데 있어 `gather`함수보다 세분화된 옵션을 지정할 수 있으며, 이를 통해 좀 더 원활한 피봇팅을 수행할 수 있습니다.

</div>

</div>

## long form -&gt; wide form

 반대의 경우도 살펴보겠습니다. 위에서 알아본 함수들은 각각 쌍으로 대응되는 함수들을 보유하고 있는데 이는 다음과 같습니다.

     - reshape2::dcast()
     - tidyr::spread()
     - tidyr::pivot_wider()

<br>

<div class="panelset">

<div class="panel">

<span class="panel-name">dcast()</span>

### reshape2::dcast()

 **`reshape2`** 패키지의 `melt` 함수와 대응되는 함수인 `dcast`함수는 데이터를 `wide form`으로 만드는데 용이한 함수입니다. 사용법으로는 다음과 같습니다.

``` r
dcast(
  data,
  formula,
  fun.aggregate = NULL,
  sep = "_",
  ...,
  margins = NULL,
  subset = NULL,
  fill = NULL,
  drop = TRUE,
  value.var = guess(data),
  verbose = getOption("datatable.verbose")
)
```

<details>
<summary>
매개변수 설명 보기
</summary>
<table class="table table-striped table-hover table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
매개변수명
</th>
<th style="text-align:left;">
의미
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;border-right:1px solid;">
data
</td>
<td style="text-align:left;">
데이터프레임 / data.table
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
formula
</td>
<td style="text-align:left;">
포뮬러
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
fun.aggregate
</td>
<td style="text-align:left;">
집계함수
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
sep
</td>
<td style="text-align:left;">
구분문자
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
fill
</td>
<td style="text-align:left;">
누락된 값 채울 값
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
drop
</td>
<td style="text-align:left;">
누락값 제외 여부
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
value.var
</td>
<td style="text-align:left;">
값 열
</td>
</tr>
</tbody>
</table>
</details>

 `gapminder_long_melt` 객체는 `wide form`형태의 데이터를 `melt`함수를 사용하여 `long form`형태로 변환했던 데이터 였습니다. `dcast`함수를 사용하여 다시 `wide form`으로 변환하는 작업을 진행해 보도록 하겠습니다.

``` r
# wide form 데이터 확인 
gapminder_wide %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>

# long form 데이터 확인 
gapminder_long_melt %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422

# wide form -> long form 변환 
gapminder_wide_dcast <- gapminder_long_melt %>% 
  reshape2::dcast(country ~ ..., value.var = "gdpPercap") %>% 
  as_tibble()

# 결과 비교 
gapminder_wide_dcast %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>

gapminder_wide %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>
```

 포뮬러상 물결무늬를 기준으로 좌측의 변수를 기준으로 나머지 열을 의미하는 `...`을 사용해 기존 변수값을 새로운 변수열로 생성하여 `wide form`데이터로의 변환을 진행하였습니다. 정상적으로 변환이 된 것을 볼 수 있었지만 이전부터 `dcast` 함수를 사용하고 있어 함수를 능숙하게 다루는 사람이 아닌 이상 `pivot wider`함수를 사용하는게 더욱 쉽고 편리할 것입니다.

<br>
<button type="button" onclick="location.href = '#long-form--gt-wide-form' ">다른 함수 보기</button>

</div>

<div class="panel">

<span class="panel-name">spread()</span>

### tidyr::spread()

 `melt`함수와 대응되는 함수인 `spread`함수는 그 명칭처럼 `long form`데이터를 `wide form`으로 넓게 펼치는 함수입니다. 함수의 사용법으로는 다음과 같습니다.

``` r
spread(data, key, value, fill = NA, convert = FALSE, drop = TRUE, sep = NULL)
```

<details>
<summary>
매개변수 설명 보기
</summary>
<table class="table table-striped table-hover table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
매개변수명
</th>
<th style="text-align:left;">
의미
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;border-right:1px solid;">
data
</td>
<td style="text-align:left;">
데이터프레임
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
key
</td>
<td style="text-align:left;">
기준 열
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
value
</td>
<td style="text-align:left;">
값 열
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
fill
</td>
<td style="text-align:left;">
결측값 대채 할 값
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
convert
</td>
<td style="text-align:left;">
값 열 클래스 자동 변환
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
drop
</td>
<td style="text-align:left;">
누락값 제외 여부
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
sep
</td>
<td style="text-align:left;">
값 열 구분 문자
</td>
</tr>
</tbody>
</table>
</details>

``` r
# wide form 데이터 확인 
gapminder_wide %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>

# long form 데이터 확인 
gapminder_long %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422

# wide form -> long form 변환 
gapminder_wide_spread <- gapminder_long %>%
  tidyr::spread(key = year, value = gdpPercap) %>% 
  as_tibble()

# 결과 비교 
gapminder_wide_spread %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>

gapminder_wide %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>
```

<br>
<button type="button" onclick="location.href = '#long-form--gt-wide-form' ">다른 함수 보기</button>

</div>

<div class="panel">

<span class="panel-name">pivot\_wider()</span>

### tidyr::pivot\_wider()

  마지막으로 알아볼 함수는 `pivot_wider`함수입니다. 마찬가지로 `pivot_longer`함수와 대응되는 함수이며, 가장 최신의 함수라 사용자 편의성에서나 기능적인 면에서나 우수한 함수라고 할 수 있습니다. `pivot_wider`함수의 사용법으로는 다음과 같습니다.

``` r
pivot_wider(
  data,
  id_cols = NULL,
  names_from = name,
  names_prefix = "",
  names_sep = "_",
  names_glue = NULL,
  names_sort = FALSE,
  names_repair = "check_unique",
  values_from = value,
  values_fill = NULL,
  values_fn = NULL,
  ...
)
```

<details>
<summary>
매개변수 설명 보기
</summary>
<table class="table table-striped table-hover table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
매개변수명
</th>
<th style="text-align:left;">
의미
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;border-right:1px solid;">
data
</td>
<td style="text-align:left;">
데이터프레임
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
id.cols
</td>
<td style="text-align:left;">
기준 열
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_from
</td>
<td style="text-align:left;">
변수 열
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_prefix
</td>
<td style="text-align:left;">
변수 열 접두어
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_sep
</td>
<td style="text-align:left;">
결합 문자
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_glue
</td>
<td style="text-align:left;">
glue문법 사용자 이름
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_sort
</td>
<td style="text-align:left;">
이름 정렬(TRUE/FALSE)
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
names\_repair
</td>
<td style="text-align:left;">
오류 발생시 처리 방식 지정
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
values\_from
</td>
<td style="text-align:left;">
값 열
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
values\_fill
</td>
<td style="text-align:left;">
결측치 채울 값
</td>
</tr>
<tr>
<td style="text-align:left;border-right:1px solid;">
values\_fn
</td>
<td style="text-align:left;">
값 적용 함수
</td>
</tr>
</tbody>
</table>
</details>

 `pivot_wider` 함수를 통해 `wide form` 데이터를 생성해 보도록 합시다.

``` r
# wide form 데이터 확인 
gapminder_wide %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>

# long form 데이터 확인 
gapminder_long %>% head(5)
#>   country year gdpPercap
#> 1 Albania 1952  1601.056
#> 2 Albania 1957  1942.284
#> 3 Albania 1962  2312.889
#> 4 Albania 1967  2760.197
#> 5 Albania 1972  3313.422

# wide form -> long form 변환 
gapminder_wide_pivot_wider <- gapminder_long %>% 
  tidyr::pivot_wider(id_cols = country, names_from = year, values_from = gdpPercap)

# 결과 비교 
gapminder_wide_pivot_wider %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>

gapminder_wide %>% head(5)
#> # A tibble: 5 x 13
#>   country  `1952` `1957` `1962` `1967` `1972` `1977` `1982` `1987` `1992` `1997`
#>   <fct>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Albania   1601.  1942.  2313.  2760.  3313.  3533.  3631.  3739.  2497.  3193.
#> 2 Austria   6137.  8843. 10751. 12835. 16662. 19749. 21597. 23688. 27042. 29096.
#> 3 Belgium   8343.  9715. 10991. 13149. 16672. 19118. 20980. 22526. 25576. 27561.
#> 4 Bosnia ~   974.  1354.  1710.  2172.  2860.  3528.  4127.  4314.  2547.  4766.
#> 5 Bulgaria  2444.  3009.  4254.  5577.  6597.  7612.  8224.  8240.  6303.  5970.
#> # ... with 2 more variables: 2002 <dbl>, 2007 <dbl>
```

 결과를 비교해 보자면 역시나 동일한 결과를 뱉는 것을 볼 수 있습니다. `spread` 함수와의 차이점이라면 `pivot_wider`함수가 좀 더 변수 열 및 값 열을 더욱 세밀하게 control 하도록 하는 옵션(인수)들을 제공한다는 점과, 그 외에도 인수들의 이름이 더욱 직관적이고 좀 더 사용자 친화적인 부분이란 점 입니다.

</div>

</div>

## 결론

  지금까지 총 세 종류의 피봇팅 함수들을 살펴 보았습니다. 예전부터 `melt`, `dcast`, `gather`, `spread` 등의 함수를 쭈욱 사용하여 해당 함수들을 쓰는데 거리낌이 없는 상황이 아니라면 피봇팅 작업을 할 시 `pivot_longer`, `pivot_wider` 함수를 사용하는 것을 추천드립니다. 이번 포스팅에서 전부 다루지는 못했지만 pivot 함수들이 다른 함수들보다 더욱 tidy한 문법에 어울리며, 또한 tidy 문법을 통해 더욱 풍부한 형 변환 및 전처리를 할 수 있기 때문입니다. 조금만 익혀둔다면 많은 도움을 받으실 수 있을거라 조심스레 생각해 봅니다.

------------------------------------------------------------------------

# 참고자료

-   [tidyr vignettes/pivot.md](https://tidyr.tidyverse.org/articles/pivot.html)

-   [tidyr vs reshape2](https://jtr13.github.io/spring19/hx2259_qz2351.html)

-   [데이터 사이언스 깔끔한 데이터(tidy data)](https://statkclee.github.io/data-science/data-handling-tidyr.html)

-   [tidyr 팩키지로 데이터프레임 솜씨있게 조작](https://statkclee.github.io/r-gapminder-kr/14-tidyr/index.html)
