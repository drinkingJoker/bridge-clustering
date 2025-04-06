const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    port: 8081,  // 明确指定前端服务端口
    proxy: {
      '/api': {
        target: 'http://localhost:8082',  // 后端SpringBoot服务地址
        changeOrigin: true,
        pathRewrite: {
          '^/api': '/api'  // 保留/api前缀以匹配后端context-path
        },
        logLevel: 'debug'  // 添加调试日志以便排查问题
      }
    }
  }
})
