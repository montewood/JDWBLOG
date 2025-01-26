
library(shiny)
# pak::pak("posit-dev/r-shinylive")
install.packages("shinylive")


library(shinylive)
getwd()

"~/JDWBLOG/themes/github.com/wowchemy/wowchemy-hugo-modules/wowchemy/layouts/partials/widgets/app"

shinylive::export(
  "~/JDWBLOG/themes/github.com/wowchemy/wowchemy-hugo-modules/wowchemy/layouts/partials/widgets/app",
  "~/JDWBLOG/themes/github.com/wowchemy/wowchemy-hugo-modules/wowchemy/layouts/partials/widgets/shinylive",
  wasm_packages = T, package_cache = F
)


?shinylive::export

install.packages("webr")

library(webr)

shinylive::assets_info()
shinylive::assets_version()
shinylive::assets_cleanup()

httpuv::runStaticServer("~/JDWBLOG/themes/github.com/wowchemy/wowchemy-hugo-modules/wowchemy/layouts/partials/widgets/shinylive/", port=8008)

