# 版本记录

## 未发布

- 算子接口统一到 `turbo_physai.operators`，原 `turbo_physai.grid_sample()` 等顶层算子入口移除；调用方需更新导入路径。
- 内部原生扩展由 `turbo_physai.ops` 更名为 `turbo_physai._C`，需重新构建并安装 wheel。
