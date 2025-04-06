package com.example.clusteringbackend.util;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

@Slf4j
@Component
public class PythonScriptRunner {

    /**
     * 执行Python脚本
     * 
     * @param command 完整的命令字符串，包括Python脚本路径和参数
     * @throws IOException 如果执行过程中发生IO异常
     * @throws Exception   如果Python脚本执行失败
     */
    public void runPythonScript(String command) throws IOException, Exception {
        log.info("执行Python命令: {}", command);

        Process process = Runtime.getRuntime().exec("python " + command);

        // 读取标准输出
        BufferedReader stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));
        // 读取错误输出
        BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));

        // 输出Python脚本的标准输出
        String s;
        while ((s = stdInput.readLine()) != null) {
            log.info("Python输出: {}", s);
        }

        // 输出Python脚本的错误输出
        StringBuilder errorMessage = new StringBuilder();
        while ((s = stdError.readLine()) != null) {
            errorMessage.append(s).append("\n");
            log.error("Python错误: {}", s);
        }

        // 等待Python脚本执行完成
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new Exception("Python脚本执行失败，退出码: " + exitCode + "\n错误信息: " + errorMessage.toString());
        }
    }
}