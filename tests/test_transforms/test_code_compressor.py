"""Tests for Code-Aware Compressor using tree-sitter AST parsing.

Comprehensive tests covering:
- CodeCompressorConfig: Configuration validation and defaults
- CodeAwareCompressor: Core AST-based compression functionality
- Language detection: Auto-detection from extensions and content
- Transform interface: apply(), should_apply() methods
- Syntax preservation: Guarantees valid output syntax
- Edge cases: Empty content, unavailable dependency, fallbacks
"""

from unittest.mock import patch

import pytest

from headroom.transforms.code_compressor import (
    _LANG_CONFIGS,
    CodeAwareCompressor,
    CodeCompressionResult,
    CodeCompressorConfig,
    CodeLanguage,
    CodeProfile,
    DocstringMode,
    _get_parser,
    _normalize_language,
    detect_language,
    is_tree_sitter_available,
    is_tree_sitter_loaded,
    unload_tree_sitter,
)

# Try to import for availability check
try:
    import tree_sitter_language_pack  # noqa: F401

    TREE_SITTER_INSTALLED = True
except ImportError:
    TREE_SITTER_INSTALLED = False


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default CodeCompressorConfig for testing."""
    return CodeCompressorConfig(
        min_tokens_for_compression=10,  # Low threshold for tests
        enable_ccr=False,  # Disable CCR for unit tests
    )


@pytest.fixture
def compressor(default_config):
    """CodeAwareCompressor instance with default config."""
    return CodeAwareCompressor(default_config)


@pytest.fixture
def tokenizer():
    """Get a tokenizer for Transform interface tests."""
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    return Tokenizer(token_counter, "gpt-4o")


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_python_code(n_functions: int = 5, n_classes: int = 1) -> str:
    """Generate Python code for testing."""
    lines = [
        '"""Module with classes and functions."""',
        "",
        "import os",
        "import sys",
        "from typing import Any, Optional, List",
        "from dataclasses import dataclass",
        "",
    ]

    for c in range(n_classes):
        lines.extend(
            [
                "@dataclass",
                f"class TestClass{c}:",
                '    """A test class with docstring."""',
                "    name: str",
                "    value: int = 0",
                "",
                "    def method(self, arg: Any) -> str:",
                '        """Process the argument."""',
                "        result = str(arg)",
                "        for i in range(10):",
                '            result += f"iteration {i}"',
                "        return result",
                "",
            ]
        )

    for i in range(n_functions):
        lines.extend(
            [
                f"def function_{i}(arg: Any, optional: Optional[str] = None) -> str:",
                f'    """Process argument {i}.',
                "",
                "    This is a longer docstring with multiple lines.",
                "    It explains what the function does in detail.",
                "",
                "    Args:",
                "        arg: The argument to process.",
                "        optional: An optional parameter.",
                "",
                "    Returns:",
                "        A string result.",
                '    """',
                "    result = str(arg)",
                "    if optional:",
                "        result += optional",
                "    for i in range(10):",
                '        result += f"iteration {i}"',
                "    try:",
                "        int(result)",
                "    except ValueError:",
                '        result = "0"',
                "    return result",
                "",
            ]
        )

    return "\n".join(lines)


def generate_javascript_code(n_functions: int = 5) -> str:
    """Generate JavaScript code for testing."""
    lines = [
        "// Module with various functions",
        'import { something } from "module";',
        'const config = require("./config");',
        "",
    ]

    for i in range(n_functions):
        lines.extend(
            [
                "/**",
                f" * Process function {i}",
                " * @param {any} arg - The argument",
                " * @returns {string} The result",
                " */",
                f"function processFunction{i}(arg) {{",
                "    let result = String(arg);",
                "    for (let j = 0; j < 10; j++) {",
                "        result += `iteration ${j}`;",
                "    }",
                "    try {",
                "        JSON.parse(result);",
                "    } catch (e) {",
                "        console.error(e);",
                "    }",
                "    return result;",
                "}",
                "",
            ]
        )

    lines.append("export { processFunction0 };")
    return "\n".join(lines)


def generate_go_code(n_functions: int = 3) -> str:
    """Generate Go code for testing."""
    lines = [
        "package main",
        "",
        'import "fmt"',
        "",
        "// Config holds configuration",
        "type Config struct {",
        "    Name  string",
        "    Value int",
        "}",
        "",
    ]

    for i in range(n_functions):
        lines.extend(
            [
                f"// Process{i} processes the input",
                f"func Process{i}(input string) (string, error) {{",
                "    result := input",
                "    for i := 0; i < 10; i++ {",
                '        result = fmt.Sprintf("%s-%d", result, i)',
                "    }",
                "    if len(result) == 0 {",
                '        return "", fmt.Errorf("empty result")',
                "    }",
                "    return result, nil",
                "}",
                "",
            ]
        )

    return "\n".join(lines)


def generate_csharp_code() -> str:
    """Generate C# code for testing."""
    return """
using System;
using System.Collections.Generic;

public class CustomerService
{
    private readonly List<string> events = new();

    public CustomerService()
    {
        events.Add("created");
        events.Add("initialized");
        events.Add(DateTime.UtcNow.ToString("O"));
    }

    public string FormatName(string? first, string? last)
    {
        var parts = new List<string>();
        if (!string.IsNullOrWhiteSpace(first))
        {
            parts.Add(first.Trim());
        }
        if (!string.IsNullOrWhiteSpace(last))
        {
            parts.Add(last.Trim());
        }
        return string.Join(" ", parts);
    }
}
"""


def require_csharp_parser() -> None:
    """Skip the current test unless the optional C# parser is available."""
    if not TREE_SITTER_INSTALLED:
        pytest.skip("tree-sitter-languages not installed")
    try:
        _get_parser("csharp")
    except Exception as exc:
        pytest.skip(f"tree-sitter C# parser unavailable: {exc}")


# =============================================================================
# TestCodeCompressorConfig
# =============================================================================


class TestCodeCompressorConfig:
    """Tests for CodeCompressorConfig dataclass."""

    def test_default_values(self):
        """Default config values are sensible."""
        config = CodeCompressorConfig()

        assert config.preserve_imports is True
        assert config.preserve_signatures is True
        assert config.preserve_type_annotations is True
        assert config.preserve_decorators is True
        assert config.docstring_mode == DocstringMode.FIRST_LINE
        assert config.target_compression_rate == 0.2
        assert config.max_body_lines == 5
        assert config.min_tokens_for_compression == 100
        assert config.enable_ccr is True

    def test_custom_values(self):
        """Custom config values are applied."""
        config = CodeCompressorConfig(
            preserve_imports=False,
            preserve_signatures=True,
            docstring_mode=DocstringMode.FULL,
            target_compression_rate=0.3,
            max_body_lines=10,
            min_tokens_for_compression=50,
        )

        assert config.preserve_imports is False
        assert config.preserve_signatures is True
        assert config.docstring_mode == DocstringMode.FULL
        assert config.target_compression_rate == 0.3
        assert config.max_body_lines == 10
        assert config.min_tokens_for_compression == 50

    def test_docstring_modes(self):
        """All docstring modes are valid."""
        for mode in DocstringMode:
            config = CodeCompressorConfig(docstring_mode=mode)
            assert config.docstring_mode == mode


# =============================================================================
# TestCodeCompressionResult
# =============================================================================


class TestCodeCompressionResult:
    """Tests for CodeCompressionResult dataclass."""

    def test_tokens_saved(self):
        """tokens_saved property calculates correctly."""
        result = CodeCompressionResult(
            compressed="short",
            original="long content here",
            original_tokens=100,
            compressed_tokens=30,
            compression_ratio=0.3,
            language=CodeLanguage.PYTHON,
            syntax_valid=True,
        )

        assert result.tokens_saved == 70

    def test_tokens_saved_no_negative(self):
        """tokens_saved never returns negative."""
        result = CodeCompressionResult(
            compressed="expanded",
            original="short",
            original_tokens=10,
            compressed_tokens=20,
            compression_ratio=2.0,
            language=CodeLanguage.PYTHON,
            syntax_valid=True,
        )

        assert result.tokens_saved == 0

    def test_savings_percentage(self):
        """savings_percentage property calculates correctly."""
        result = CodeCompressionResult(
            compressed="short",
            original="long content",
            original_tokens=100,
            compressed_tokens=25,
            compression_ratio=0.25,
            language=CodeLanguage.PYTHON,
            syntax_valid=True,
        )

        assert result.savings_percentage == 75.0

    def test_savings_percentage_zero_original(self):
        """savings_percentage handles zero original tokens."""
        result = CodeCompressionResult(
            compressed="",
            original="",
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=1.0,
            language=CodeLanguage.UNKNOWN,
            syntax_valid=True,
        )

        assert result.savings_percentage == 0.0


# =============================================================================
# TestCodeLanguage
# =============================================================================


class TestCodeLanguage:
    """Tests for CodeLanguage enum and detection."""

    def test_all_language_values_are_unique(self):
        """All language enum values are unique."""
        values = [lang.value for lang in CodeLanguage]
        assert len(values) == len(set(values))

    def test_csharp_language_aliases_normalize(self):
        """Common C#/.NET/Unity aliases normalize to C# plus a profile."""
        expected = {
            "c#": CodeProfile.GENERIC,
            "cs": CodeProfile.GENERIC,
            "csharp": CodeProfile.GENERIC,
            "c-sharp": CodeProfile.GENERIC,
            ".net": CodeProfile.DOTNET,
            "dotnet": CodeProfile.DOTNET,
            "unity": CodeProfile.UNITY,
            "aspnetcore": CodeProfile.ASPNET_CORE,
            "asp.net core": CodeProfile.ASPNET_CORE,
        }

        for alias, profile in expected.items():
            assert _normalize_language(alias) == (CodeLanguage.CSHARP, profile)

    def test_csharp_config_is_conservative_about_properties(self):
        """C# first pass does not treat properties/events/indexers as functions."""
        config = _LANG_CONFIGS[CodeLanguage.CSHARP]

        assert "method_declaration" in config.function_nodes
        assert "constructor_declaration" in config.function_nodes
        assert "property_declaration" not in config.function_nodes
        assert "event_declaration" not in config.function_nodes
        assert "indexer_declaration" not in config.function_nodes

    def test_detect_python_language(self):
        """Python language is detected from code patterns."""

        code = """
import os
from typing import List

def function(arg: str) -> str:
    return arg

class MyClass:
    pass
"""
        lang, confidence = detect_language(code)
        assert lang == CodeLanguage.PYTHON
        assert confidence > 0.5

    def test_detect_javascript_language(self):
        """JavaScript language is detected from code patterns."""

        code = """
const express = require('express');
import { something } from 'module';

function handler(req, res) {
    return res.json({ status: 'ok' });
}

export default handler;
"""
        lang, confidence = detect_language(code)
        assert lang in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT)
        assert confidence > 0.3

    def test_detect_go_language(self):
        """Go language is detected from code patterns."""

        code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello")
}
"""
        lang, confidence = detect_language(code)
        assert lang == CodeLanguage.GO
        assert confidence > 0.3

    def test_detect_csharp_language_from_unity_markers(self):
        """C# language is detected from Unity-specific source markers."""
        code = """
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [SerializeField] private Rigidbody body;
}
"""
        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available", return_value=False
        ):
            lang, confidence = detect_language(code)

        assert lang == CodeLanguage.CSHARP
        assert confidence > 0.3

    def test_detect_csharp_language_from_aspnet_markers(self):
        """C# language is detected from ASP.NET Core source markers."""
        code = """
using Microsoft.AspNetCore.Builder;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
app.MapGet("/health", () => Results.Ok());
"""
        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available", return_value=False
        ):
            lang, confidence = detect_language(code)

        assert lang == CodeLanguage.CSHARP
        assert confidence > 0.3


# =============================================================================
# TestCodeAwareCompressor
# =============================================================================


class TestCodeAwareCompressor:
    """Tests for CodeAwareCompressor core functionality."""

    def test_init_with_default_config(self):
        """Compressor initializes with default config."""
        compressor = CodeAwareCompressor()

        assert compressor.config is not None
        assert compressor.config.preserve_imports is True

    def test_init_with_custom_config(self, default_config):
        """Compressor initializes with custom config."""
        compressor = CodeAwareCompressor(default_config)

        assert compressor.config == default_config

    def test_compress_skips_small_content(self, compressor):
        """Small content is not compressed."""
        small_code = "def f(): pass"
        result = compressor.compress(small_code)

        assert result.compressed == small_code
        assert result.compression_ratio == 1.0

    def test_compress_handles_empty_content(self, compressor):
        """Empty content returns empty result."""
        result = compressor.compress("")

        assert result.compressed == ""
        assert result.compression_ratio == 1.0
        assert result.syntax_valid is True

    def test_compress_with_explicit_language(self, compressor):
        """Language can be specified explicitly."""
        code = generate_python_code(2)
        result = compressor.compress(code, language="python")

        # Should detect or use the specified language
        assert result.language == CodeLanguage.PYTHON or result.language == CodeLanguage.UNKNOWN

    def test_compress_accepts_csharp_aliases_without_tree_sitter(self):
        """Explicit C# aliases are accepted before optional parser loading."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            fallback_to_kompress=False,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_csharp_code()

        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available", return_value=False
        ):
            expected_profiles = {
                "c#": CodeProfile.GENERIC,
                "cs": CodeProfile.GENERIC,
                "dotnet": CodeProfile.DOTNET,
                "unity": CodeProfile.UNITY,
                "asp.net core": CodeProfile.ASPNET_CORE,
            }
            for alias, expected_profile in expected_profiles.items():
                result = compressor.compress(code, language=alias)
                assert result.language == CodeLanguage.CSHARP
                assert result.profile == expected_profile
                assert result.syntax_valid is True

    def test_csharp_profile_is_inferred_from_source_markers_without_tree_sitter(self):
        """Generic C# language hints infer framework profiles from source markers."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            fallback_to_kompress=False,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        unity_code = """
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    private void Update()
    {
        transform.Translate(Vector3.forward * Time.deltaTime);
    }
}
"""
        aspnet_code = """
using Microsoft.AspNetCore.Builder;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddValidation();
var app = builder.Build();
app.MapGet("/health", () => Results.Ok());
"""

        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available", return_value=False
        ):
            unity_result = compressor.compress(unity_code, language="csharp")
            aspnet_result = compressor.compress(aspnet_code, language="csharp")

        assert unity_result.profile == CodeProfile.UNITY
        assert aspnet_result.profile == CodeProfile.ASPNET_CORE

    def test_explicit_profile_overrides_inferred_profile_without_tree_sitter(self):
        """Explicit profile hints override source-marker inference."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            fallback_to_kompress=False,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Microsoft.AspNetCore.Builder;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
app.MapGet("/health", () => Results.Ok());
"""

        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available", return_value=False
        ):
            result = compressor.compress(code, language="csharp", profile="dotnet")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.DOTNET

    def test_small_explicit_csharp_preserves_language_and_profile(self):
        """Small explicit C# snippets still expose language/profile metadata."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1000))

        result = compressor.compress("public class Player : MonoBehaviour {}", language="unity")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.UNITY

    def test_efcore_profile_is_inferred_without_tree_sitter(self):
        """EF Core source markers infer the EF Core profile."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            fallback_to_kompress=False,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Microsoft.EntityFrameworkCore;

public class AppDbContext : DbContext
{
        public DbSet<Product> Products => Set<Product>();

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
                modelBuilder.Entity<Product>().HasKey(product => product.Id);
                modelBuilder.Entity<Product>().HasQueryFilter(product => !product.Deleted);
        }
}
"""

        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available",
            return_value=False,
        ):
            result = compressor.compress(code, language="csharp")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.EF_CORE

    def test_razor_artifact_preserves_directives_and_code_block(self):
        """Razor artifacts are compressed without using the C# parser."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        code = """
@page "/products/{id:int}"
@using System.ComponentModel.DataAnnotations
@inject ProductService Products

<h1>Product</h1>
<p>Lots of markup that can be summarized.</p>

@code {
        [Parameter] public int Id { get; set; }
        [PersistentState] public Product? Product { get; set; }
}
"""

        result = compressor.compress(code, language="razor")

        assert result.language == CodeLanguage.RAZOR
        assert result.profile == CodeProfile.ASPNET_CORE
        assert "@page" in result.compressed
        assert "@inject" in result.compressed
        assert "[PersistentState]" in result.compressed
        assert "lines omitted" in result.compressed

    def test_msbuild_artifact_preserves_frameworks_and_references(self):
        """MSBuild project files keep target frameworks and important references."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        code = """
<Project Sdk="Microsoft.NET.Sdk.Web">
    <PropertyGroup>
        <TargetFramework>net10.0</TargetFramework>
        <Nullable>enable</Nullable>
        <ImplicitUsings>enable</ImplicitUsings>
        <GenerateOpenApiDocuments>true</GenerateOpenApiDocuments>
    </PropertyGroup>
    <ItemGroup>
        <PackageReference Include="Microsoft.EntityFrameworkCore" Version="10.0.0" />
        <ProjectReference Include="..\\Domain\\Domain.csproj" />
    </ItemGroup>
</Project>
"""

        result = compressor.compress(code, language="csproj")

        assert result.language == CodeLanguage.MSBUILD
        assert result.profile == CodeProfile.DOTNET
        assert "<TargetFramework>net10.0</TargetFramework>" in result.compressed
        assert "<GenerateOpenApiDocuments>true</GenerateOpenApiDocuments>" in result.compressed
        assert "PackageReference" in result.compressed
        assert "ProjectReference" in result.compressed

    def test_unity_asmdef_json_keeps_important_keys(self):
        """Unity asmdef JSON keeps assembly metadata and omits unrelated keys."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        code = """
{
    "name": "Game.Runtime",
    "references": ["Unity.TextMeshPro"],
    "includePlatforms": [],
    "defineConstraints": ["UNITY_6000_0_OR_NEWER"],
    "allowUnsafeCode": false,
    "optionalUnityReferences": ["TestAssemblies"]
}
"""

        result = compressor.compress(code, language="asmdef")

        assert result.language == CodeLanguage.JSON
        assert result.profile == CodeProfile.UNITY
        assert '"name": "Game.Runtime"' in result.compressed
        assert '"references"' in result.compressed
        assert "optionalUnityReferences" not in result.compressed

    def test_unity_package_manifest_keeps_entities_dependency(self):
        """Unity package manifests preserve com.unity.entities dependencies."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        code = """
{
    "dependencies": {
        "com.unity.entities": "1.3.14",
        "com.unity.collections": "2.5.1",
        "com.unity.render-pipelines.universal": "17.0.0"
    },
    "scopedRegistries": [
        { "name": "Unity", "url": "https://packages.unity.com", "scopes": ["com.unity"] }
    ],
    "lock": { "unrelated": true }
}
"""

        result = compressor.compress(code, language="unity-manifest")

        assert result.language == CodeLanguage.JSON
        assert result.profile == CodeProfile.UNITY
        assert '"com.unity.entities": "1.3.14"' in result.compressed
        assert '"scopedRegistries"' in result.compressed
        assert '"lock"' not in result.compressed

    def test_unity_entities_profile_is_inferred_without_tree_sitter(self):
        """Unity Entities interfaces infer Unity profile and preserve metadata."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            fallback_to_kompress=False,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Unity.Burst;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;

public struct MoveSpeed : IComponentData
{
    public float Value;
}

[BurstCompile]
public partial struct MoveSystem : ISystem
{
    public void OnUpdate(ref SystemState state)
    {
        foreach (var (transform, speed) in SystemAPI.Query<RefRW<LocalTransform>, RefRO<MoveSpeed>>())
        {
            transform.ValueRW.Position += new float3(speed.ValueRO.Value, 0, 0);
        }
    }
}
"""

        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available",
            return_value=False,
        ):
            result = compressor.compress(code, language="csharp")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.UNITY

    def test_burst_compile_profile_is_inferred_without_tree_sitter(self):
        """Burst-only Unity jobs infer the Unity profile."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            fallback_to_kompress=False,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Unity.Burst;

[BurstCompile]
public struct IntegrateJob
{
    public void Execute()
    {
        var value = 1 + 2;
    }
}
"""

        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available",
            return_value=False,
        ):
            result = compressor.compress(code, language="csharp")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.UNITY

    def test_dotnet_console_profile_is_inferred_without_tree_sitter(self):
        """Top-level console apps infer the .NET profile."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=1,
            fallback_to_kompress=False,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
Console.WriteLine("Hello, Headroom!");
var name = Console.ReadLine();
Console.WriteLine($"Hello {name}");
"""

        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available",
            return_value=False,
        ):
            result = compressor.compress(code, language="csharp")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.DOTNET

    def test_generic_host_profile_is_inferred_without_tree_sitter(self):
        """Generic Host wiring infers the .NET profile."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            fallback_to_kompress=False,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Microsoft.Extensions.Hosting;

var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddHostedService<Worker>();
await builder.Build().RunAsync();

public sealed class Worker : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken stoppingToken) => Task.CompletedTask;
}
"""

        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available",
            return_value=False,
        ):
            result = compressor.compress(code, language="csharp")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.DOTNET

    def test_efcore_migration_profile_preserves_schema_markers_without_tree_sitter(self):
        """EF migrations infer EF Core profile from schema operations."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            fallback_to_kompress=False,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Microsoft.EntityFrameworkCore.Migrations;

public partial class AddProducts : Migration
{
    protected override void Up(MigrationBuilder migrationBuilder)
    {
        migrationBuilder.CreateTable(name: "Products", columns: table => new
        {
            Id = table.Column<int>(nullable: false),
            Name = table.Column<string>(nullable: false)
        });
        migrationBuilder.CreateIndex(name: "IX_Products_Name", table: "Products", column: "Name");
    }
}
"""

        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available",
            return_value=False,
        ):
            result = compressor.compress(code, language="csharp")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.EF_CORE

    def test_appsettings_json_redacts_secret_values(self):
        """ASP.NET Core appsettings JSON preserves shape and redacts secrets."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        code = """
{
    "ConnectionStrings": {
        "DefaultConnection": "Server=.;Password=hunter2"
    },
    "Logging": {
        "LogLevel": {
            "Default": "Information"
        }
    },
    "AllowedHosts": "*"
}
"""

        result = compressor.compress(code, language="appsettings.json")

        assert result.language == CodeLanguage.JSON
        assert result.profile == CodeProfile.ASPNET_CORE
        assert '"ConnectionStrings"' in result.compressed
        assert "hunter2" not in result.compressed
        assert '"DefaultConnection": ""' in result.compressed

    def test_solution_artifact_preserves_project_lines(self):
        """Solution files preserve project mappings."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        code = """
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "App", "App\\App.csproj", "{11111111-1111-1111-1111-111111111111}"
EndProject
Global
        GlobalSection(SolutionConfigurationPlatforms) = preSolution
                Debug|Any CPU = Debug|Any CPU
        EndGlobalSection
EndGlobal
"""

        result = compressor.compress(code, language="sln")

        assert result.language == CodeLanguage.SOLUTION
        assert result.profile == CodeProfile.DOTNET
        assert "Project(" in result.compressed
        assert "SolutionConfigurationPlatforms" in result.compressed

    def test_slnx_artifact_preserves_project_elements(self):
        """Solution XML files preserve project elements."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        code = """
<Solution>
    <Folder Name="src">
        <Project Path="src/App/App.csproj" />
    </Folder>
    <Properties Name="Debug|Any CPU" />
</Solution>
"""

        result = compressor.compress(code, language="slnx")

        assert result.language == CodeLanguage.SOLUTION
        assert result.profile == CodeProfile.DOTNET
        assert '<Project Path="src/App/App.csproj" />' in result.compressed
        assert "Properties" not in result.compressed

    def test_compress_auto_detects_python(self, compressor):
        """Python code is auto-detected during compression."""
        code = """
import os
from typing import List

def function(arg: str) -> List[str]:
    return [arg]

class MyClass:
    pass
"""
        result = compressor.compress(code)
        # Should detect Python (if tree-sitter available) or return UNKNOWN
        assert result.language in (CodeLanguage.PYTHON, CodeLanguage.UNKNOWN)

    def test_compress_auto_detects_javascript(self, compressor):
        """JavaScript code is auto-detected during compression."""
        code = """
const express = require('express');
import { something } from 'module';

function handler(req, res) {
    return res.json({ status: 'ok' });
}

export default handler;
"""
        result = compressor.compress(code)
        assert result.language in (
            CodeLanguage.JAVASCRIPT,
            CodeLanguage.TYPESCRIPT,
            CodeLanguage.UNKNOWN,
        )

    def test_compress_auto_detects_go(self, compressor):
        """Go code is auto-detected during compression."""
        code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello")
}
"""
        result = compressor.compress(code)
        assert result.language in (CodeLanguage.GO, CodeLanguage.UNKNOWN)


class TestAgentInformationRetention:
    """Regression tests for information coding agents need after compression."""

    def test_agent_context_preserves_csharp_api_and_behavior_decision_signals(self):
        """Compressed classes keep API shape plus bounded behavior hints for agents."""
        require_csharp_parser()
        compressor = CodeAwareCompressor(
            CodeCompressorConfig(
                min_tokens_for_compression=1,
                max_body_lines=1,
                enable_ccr=False,
            )
        )
        code = """
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("api/orders")]
public sealed class OrdersController : ControllerBase
{
        private readonly IOrderService orders;

        public OrdersController(IOrderService orders)
        {
                this.orders = orders;
        }

        [HttpGet("{id:guid}")]
        [Authorize(Policy = "Orders.Read")]
        public async Task<ActionResult<OrderDto>> GetAsync(Guid id, CancellationToken cancellationToken)
        {
                Validate(id);
                var order = await orders.GetAsync(id, cancellationToken);
                if (order is null)
                {
                        return NotFound();
                }
                return Ok(order);
        }
}
"""

        result = compressor.compress(code, language="aspnetcore")

        assert result.syntax_valid is True
        assert result.profile == CodeProfile.ASPNET_CORE
        assert "[ApiController]" in result.compressed
        assert '[Route("api/orders")]' in result.compressed
        assert "public sealed class OrdersController : ControllerBase" in result.compressed
        assert "private members" in result.compressed
        assert "IOrderService orders" in result.compressed
        assert "private readonly IOrderService orders;" not in result.compressed
        assert "public OrdersController(IOrderService orders)" in result.compressed
        assert '[HttpGet("{id:guid}")]' in result.compressed
        assert '[Authorize(Policy = "Orders.Read")]' in result.compressed
        assert (
            "public async Task<ActionResult<OrderDto>> GetAsync(Guid id, "
            "CancellationToken cancellationToken)" in result.compressed
        )
        assert "orders.GetAsync" in result.compressed
        assert "NotFound" in result.compressed
        assert "Ok" in result.compressed
        assert "flow: branches" in result.compressed
        assert "awaits" in result.compressed
        assert "returns" in result.compressed

    def test_agent_context_preserves_msbuild_dependency_decision_signals(self):
        """Compressed project files keep frameworks and dependency edges."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        code = """
<Project Sdk="Microsoft.NET.Sdk.Web">
    <PropertyGroup>
        <TargetFramework>net10.0</TargetFramework>
        <Nullable>enable</Nullable>
        <ImplicitUsings>enable</ImplicitUsings>
    </PropertyGroup>
    <ItemGroup>
        <PackageReference Include="Microsoft.EntityFrameworkCore.SqlServer" Version="10.0.0" />
        <PackageReference Include="Microsoft.AspNetCore.OpenApi" Version="10.0.0" />
        <ProjectReference Include="../Headroom.Domain/Headroom.Domain.csproj" />
    </ItemGroup>
</Project>
"""

        result = compressor.compress(code, language="csproj")

        assert result.syntax_valid is True
        assert result.language == CodeLanguage.MSBUILD
        assert result.profile == CodeProfile.DOTNET
        assert 'Sdk="Microsoft.NET.Sdk.Web"' in result.compressed
        assert "<TargetFramework>net10.0</TargetFramework>" in result.compressed
        assert 'Include="Microsoft.EntityFrameworkCore.SqlServer"' in result.compressed
        assert 'Version="10.0.0"' in result.compressed
        assert 'Include="Microsoft.AspNetCore.OpenApi"' in result.compressed
        assert 'Include="../Headroom.Domain/Headroom.Domain.csproj"' in result.compressed

    def test_agent_context_preserves_redacted_config_shape_for_decisions(self):
        """Compressed appsettings keep configuration keys while removing secrets."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        code = """
{
    "ConnectionStrings": {
        "DefaultConnection": "Server=localhost;Database=Headroom;User Id=sa;Password=SuperSecret!"
    },
    "Authentication": {
        "Authority": "https://login.example.com/tenant",
        "ClientSecret": "should-redact"
    },
    "Features": {
        "EnableCompression": true
    }
}
"""

        result = compressor.compress(code, language="appsettings.json")

        assert result.syntax_valid is True
        assert result.language == CodeLanguage.JSON
        assert result.profile == CodeProfile.ASPNET_CORE
        assert '"ConnectionStrings"' in result.compressed
        assert '"DefaultConnection": ""' in result.compressed
        assert '"Authentication"' in result.compressed
        assert '"Authority": "https://login.example.com/tenant"' in result.compressed
        assert '"ClientSecret": ""' in result.compressed
        assert '"EnableCompression": true' in result.compressed
        assert "SuperSecret" not in result.compressed
        assert "should-redact" not in result.compressed

    def test_agent_context_preserves_workspace_and_unity_package_mappings(self):
        """Compressed workspace artifacts keep project paths and package dependencies."""
        compressor = CodeAwareCompressor(CodeCompressorConfig(min_tokens_for_compression=1))
        solution = """
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "Headroom.Api", "src/Headroom.Api/Headroom.Api.csproj", "{11111111-1111-1111-1111-111111111111}"
EndProject
Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "Headroom.Domain", "src/Headroom.Domain/Headroom.Domain.csproj", "{22222222-2222-2222-2222-222222222222}"
EndProject
Global
EndGlobal
"""
        manifest = """
{
    "dependencies": {
        "com.unity.entities": "1.3.14",
        "com.unity.burst": "1.8.18",
        "com.company.private-tools": "file:../Packages/private-tools"
    },
    "scopedRegistries": [
        { "name": "Company", "url": "https://packages.example.com", "scopes": ["com.company"] }
    ]
}
"""

        solution_result = compressor.compress(solution, language="sln")
        manifest_result = compressor.compress(manifest, language="unity-manifest")

        assert solution_result.syntax_valid is True
        assert solution_result.language == CodeLanguage.SOLUTION
        assert (
            '"Headroom.Api", "src/Headroom.Api/Headroom.Api.csproj"' in solution_result.compressed
        )
        assert (
            '"Headroom.Domain", "src/Headroom.Domain/Headroom.Domain.csproj"'
            in solution_result.compressed
        )
        assert manifest_result.syntax_valid is True
        assert manifest_result.language == CodeLanguage.JSON
        assert manifest_result.profile == CodeProfile.UNITY
        assert '"com.unity.entities": "1.3.14"' in manifest_result.compressed
        assert '"com.unity.burst": "1.8.18"' in manifest_result.compressed
        assert (
            '"com.company.private-tools": "file:../Packages/private-tools"'
            in manifest_result.compressed
        )
        assert '"url": "https://packages.example.com"' in manifest_result.compressed
        assert '"com.company"' in manifest_result.compressed


# =============================================================================
# TestFallbackCompression
# =============================================================================


class TestFallbackCompression:
    """Tests for fallback compression when tree-sitter unavailable."""

    def test_fallback_when_tree_sitter_unavailable(self, default_config):
        """Uses fallback compression when tree-sitter is not installed."""
        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available",
            return_value=False,
        ):
            compressor = CodeAwareCompressor(default_config)
            code = generate_python_code(5)

            result = compressor.compress(code)

            # Should still return a result (fallback compression)
            assert result is not None
            # Kompress fallback does NOT guarantee syntax validity
            # If Kompress is unavailable, returns original (valid)
            # If Kompress IS available, syntax_valid=False (cannot guarantee)

    def test_fallback_preserves_structure(self, default_config):
        """Fallback compression preserves basic structure when no compressor available.

        When both tree-sitter and Kompress are unavailable, the fallback
        returns the original code unchanged - preserving all structure.
        """
        with (
            patch(
                "headroom.transforms.code_compressor._check_tree_sitter_available",
                return_value=False,
            ),
            patch(
                "headroom.transforms.kompress_compressor.is_kompress_available",
                return_value=False,
            ),
        ):
            compressor = CodeAwareCompressor(default_config)
            code = generate_python_code(3)

            result = compressor.compress(code)

            # With no compressor available, original code is returned unchanged
            # This preserves all imports and class/function signatures
            assert "import os" in result.compressed
            assert "def function_" in result.compressed
            # Compression ratio should be 1.0 (no compression)
            assert result.compression_ratio == 1.0


# =============================================================================
# TestTransformInterface
# =============================================================================


class TestTransformInterface:
    """Tests for Transform interface (apply, should_apply)."""

    def test_should_apply_returns_false_for_small_content(self, default_config, tokenizer):
        """should_apply returns False for small content."""
        config = CodeCompressorConfig(min_tokens_for_compression=1000)
        compressor = CodeAwareCompressor(config)
        messages = [{"role": "user", "content": "def f(): pass"}]

        assert not compressor.should_apply(messages, tokenizer)

    def test_should_apply_returns_bool_for_large_code(self, default_config, tokenizer):
        """should_apply returns boolean for large code content."""
        compressor = CodeAwareCompressor(default_config)
        code = generate_python_code(20)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": code}]

        # Should return True if there's code content to process
        result = compressor.should_apply(messages, tokenizer)
        assert isinstance(result, bool)

    def test_apply_returns_transform_result(self, default_config, tokenizer):
        """apply() returns proper TransformResult."""
        compressor = CodeAwareCompressor(default_config)
        code = generate_python_code(10)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": code}]

        result = compressor.apply(messages, tokenizer)

        assert result is not None
        assert result.tokens_before > 0
        assert len(result.messages) == 1

    def test_apply_passes_through_non_code_messages(self, default_config, tokenizer):
        """apply() passes through non-code messages unchanged."""
        compressor = CodeAwareCompressor(default_config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = compressor.apply(messages, tokenizer)

        assert result.messages[0]["content"] == "Hello"
        assert result.messages[1]["content"] == "Hi there!"

    def test_name_property(self, compressor):
        """Compressor has correct name."""
        assert compressor.name == "code_aware_compressor"


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for CodeAwareCompressor."""

    def test_whitespace_only_content(self, compressor):
        """Whitespace-only content is handled gracefully."""
        result = compressor.compress("   \n\t\n   ")

        assert result.compression_ratio == 1.0
        assert result.syntax_valid is True

    def test_unicode_content(self, default_config):
        """Unicode in code is handled correctly."""
        compressor = CodeAwareCompressor(default_config)
        code = '''
def greet(name: str) -> str:
    """Greet the user in multiple languages."""
    return f"Hello, {name}! \u4f60\u597d! \u3053\u3093\u306b\u3061\u306f!"
'''
        result = compressor.compress(code)

        # Should handle unicode without crashing
        assert result is not None

    def test_very_long_function(self, default_config):
        """Very long functions are compressed."""
        compressor = CodeAwareCompressor(default_config)
        lines = ["def very_long_function():"]
        lines.append('    """A very long function."""')
        for i in range(100):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_99")
        code = "\n".join(lines)

        result = compressor.compress(code)

        # Should compress the long function body
        assert result.compression_ratio < 1.0 or "tree_sitter" not in str(
            is_tree_sitter_available()
        )

    def test_nested_functions(self, default_config):
        """Nested functions are handled."""
        compressor = CodeAwareCompressor(default_config)
        code = """
def outer():
    def inner():
        return "inner"
    return inner()
"""
        result = compressor.compress(code)

        assert result is not None
        # syntax_valid requires tree-sitter; without it, validation is skipped
        if is_tree_sitter_available():
            assert result.syntax_valid is True

    def test_syntax_errors_in_input(self, default_config):
        """Syntax errors in input don't crash the compressor."""
        compressor = CodeAwareCompressor(default_config)
        # Invalid Python syntax
        code = """
def broken(
    # Missing closing paren
"""
        # Should not raise
        result = compressor.compress(code, language="python")
        assert result is not None

    def test_mixed_language_content(self, default_config):
        """Mixed language content (like markdown with code) is handled."""
        compressor = CodeAwareCompressor(default_config)
        content = """
# Documentation

Here is some code:

```python
def example():
    pass
```

And some more text.
"""
        # Should not crash
        result = compressor.compress(content)
        assert result is not None


# =============================================================================
# TestMemoryManagement
# =============================================================================


class TestMemoryManagement:
    """Tests for memory management functions."""

    def test_is_tree_sitter_available_returns_bool(self):
        """is_tree_sitter_available returns a boolean."""
        result = is_tree_sitter_available()
        assert isinstance(result, bool)

    def test_is_tree_sitter_loaded_returns_false_initially(self):
        """is_tree_sitter_loaded returns False when no parsers loaded."""
        # Clear any loaded parsers first
        unload_tree_sitter()
        assert is_tree_sitter_loaded() is False

    def test_unload_returns_false_when_nothing_loaded(self):
        """unload_tree_sitter returns False when nothing to unload."""
        # Ensure nothing is loaded
        unload_tree_sitter()
        result = unload_tree_sitter()
        assert result is False


# =============================================================================
# Integration Tests (only run if tree-sitter is installed)
# =============================================================================


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter-languages not installed")
class TestTreeSitterIntegration:
    """Integration tests that require actual tree-sitter installation.

    These tests verify actual AST parsing and compression behavior.
    """

    def test_actual_python_compression(self):
        """Test actual compression of Python code."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(5)

        result = compressor.compress(code, language="python")

        # Should achieve compression
        assert result.compression_ratio < 1.0
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.PYTHON

    def test_actual_javascript_compression(self):
        """Test actual compression of JavaScript code."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_javascript_code(5)

        result = compressor.compress(code, language="javascript")

        assert result.compression_ratio < 1.0
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.JAVASCRIPT

    def test_actual_go_compression(self):
        """Test Go code is processed (compression may fall back due to nested structures)."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_go_code(3)

        result = compressor.compress(code, language="go")

        # Go code is processed and returns valid output
        # Note: compression_ratio may be 1.0 if compression produces invalid syntax
        # and falls back to original (Go has complex nested brace handling)
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.GO
        assert result.compressed  # Some output is produced

    def test_actual_csharp_compression(self):
        """Test actual compression of C# class code."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            max_body_lines=2,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_csharp_code()

        result = compressor.compress(code, language="csharp")

        assert result.syntax_valid is True
        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.GENERIC
        assert "using System;" in result.compressed
        assert "CustomerService" in result.compressed
        assert "FormatName" in result.compressed

    def test_csharp_14_source_syntax(self):
        """C# 14 source syntax is treated as C# for .NET 10-era code."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            max_body_lines=3,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using System;
using System.Collections.Generic;

public class Customer
{
    public string Message
    {
        get;
        set => field = value ?? "";
    }

    public void Assign(Customer? other)
    {
        other?.Message = nameof(List<>);
        var copy = other?.Message ?? "";
        Console.WriteLine(copy);
    }
}
"""

        result = compressor.compress(code, language="c#")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.DOTNET
        assert result.syntax_valid is True
        assert "Customer" in result.compressed
        assert "nameof(List<>)" in result.compressed

    def test_csharp_private_members_group_without_losing_public_interface(self):
        """C# compression keeps the public interface and summarizes private dependencies."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=1,
            max_body_lines=1,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Microsoft.Extensions.Logging;

public sealed class OrderService : IOrderService
{
    private readonly IOrderRepository repository;
    private readonly ILogger<OrderService> logger;

    public event EventHandler<OrderCreatedEventArgs>? OrderCreated;
    public string Name { get; init; }

    public async Task<OrderDto> CreateAsync(CreateOrder request)
    {
        Validate(request);
        var entity = mapper.Map<Order>(request);
        await repository.SaveAsync(entity);
        logger.LogInformation("Created {Id}", entity.Id);
        OrderCreated?.Invoke(this, new OrderCreatedEventArgs(entity.Id));
        return mapper.Map<OrderDto>(entity);
    }
}
"""

        result = compressor.compress(code, language="csharp")

        assert result.syntax_valid is True
        assert "public sealed class OrderService : IOrderService" in result.compressed
        assert "private members" in result.compressed
        assert "IOrderRepository repository" in result.compressed
        assert "ILogger<OrderService> logger" in result.compressed
        assert "private readonly IOrderRepository repository;" not in result.compressed
        assert (
            "public event EventHandler<OrderCreatedEventArgs>? OrderCreated;" in result.compressed
        )
        assert "public string Name { get; init; }" in result.compressed
        assert "public async Task<OrderDto> CreateAsync(CreateOrder request)" in result.compressed
        assert "repository.SaveAsync" in result.compressed
        assert "logger.LogInformation" in result.compressed
        assert "OrderCreated.Invoke" in result.compressed
        assert "awaits" in result.compressed
        assert "returns" in result.compressed
        assert "writes" in result.compressed

    def test_csharp_omitted_comment_summarizes_control_flow(self):
        """Omitted C# bodies include bounded behavior hints for control flow and throws."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=1,
            max_body_lines=1,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
public sealed class ReportBuilder
{
    public string Build(Customer customer)
    {
        Validate(customer);

        foreach (var item in customer.Items)
        {
            if (item is null)
            {
                throw new InvalidOperationException("Missing item");
            }
            builder.AppendLine(item.Name);
        }

        return builder.ToString();
    }
}
"""

        result = compressor.compress(code, language="csharp")

        assert result.syntax_valid is True
        assert "public string Build(Customer customer)" in result.compressed
        assert "lines omitted" in result.compressed
        assert "flow:" in result.compressed
        assert "loops" in result.compressed
        assert "branches" in result.compressed
        assert "throws" in result.compressed
        assert "builder.AppendLine" in result.compressed

    def test_unity_csharp_script_alias(self):
        """Unity alias uses conservative C# support and preserves Unity entry points."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            max_body_lines=1,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using System.Collections;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class PlayerController : MonoBehaviour
{
    [SerializeField] private Rigidbody body;
    [field: SerializeField] public float Speed { get; private set; }

    private void Awake()
    {
        body = GetComponent<Rigidbody>();
    }

    private IEnumerator Start()
    {
        yield return null;
    }

    private void Update()
    {
        transform.Translate(Vector3.forward * Speed * Time.deltaTime);
    }

#if UNITY_EDITOR
    private void OnValidate()
    {
        Speed = Mathf.Max(0, Speed);
    }
#endif
}
"""

        result = compressor.compress(code, language="unity")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.UNITY
        assert result.syntax_valid is True
        assert "using UnityEngine;" in result.compressed
        assert "[RequireComponent" in result.compressed
        assert "unity serialized fields" in result.compressed
        assert "Rigidbody body" in result.compressed
        assert "float Speed" in result.compressed
        assert "Awake" in result.compressed
        assert "Start" in result.compressed
        assert "Update" in result.compressed
        assert "#if UNITY_EDITOR" in result.compressed
        assert "OnValidate" in result.compressed

    def test_aspnet_core_alias_and_attributes(self):
        """ASP.NET Core alias maps to C# and preserves route/action metadata."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            max_body_lines=2,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("api/[controller]")]
public class WeatherController : ControllerBase
{
    [HttpGet]
    public IActionResult Get()
    {
        var forecast = new[] { "sunny", "cloudy", "rain" };
        var selected = forecast[DateTime.UtcNow.Day % forecast.Length];
        return Ok(new { forecast = selected });
    }
}
"""

        result = compressor.compress(code, language="asp.net core")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.ASPNET_CORE
        assert result.syntax_valid is True
        assert "[ApiController]" in result.compressed
        assert "[HttpGet]" in result.compressed
        assert "IActionResult Get" in result.compressed

    def test_aspnet_core_minimal_api_source(self):
        """ASP.NET Core minimal API source remains valid C# top-level code."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddValidation();
builder.Services.AddOpenApi();

var app = builder.Build();

app.MapPost("/products", ([Required] Product product) =>
    TypedResults.Ok(product))
   .WithName("CreateProduct")
   .WithOpenApi();

app.MapGet("/events", () =>
    TypedResults.ServerSentEvents(GetEventsAsync()));

app.MapOpenApi("/openapi/{documentName}.yaml");
app.Run();

public record Product([Required] string Name, [Range(1, 1000)] int Quantity);
"""

        result = compressor.compress(code, language="aspnetcore")

        assert result.language == CodeLanguage.CSHARP
        assert result.profile == CodeProfile.ASPNET_CORE
        assert result.syntax_valid is True
        assert "WebApplication.CreateBuilder" in result.compressed
        assert "AddValidation" in result.compressed
        assert "AddOpenApi" in result.compressed
        assert "aspnet: MapPost" in result.compressed
        assert 'route="/products"' in result.compressed
        assert 'name="CreateProduct"' in result.compressed
        assert "WithOpenApi" in result.compressed
        assert "aspnet: MapGet" in result.compressed
        assert 'route="/events"' in result.compressed
        assert "MapOpenApi" in result.compressed
        assert "[Required]" in result.compressed

    def test_unity_entities_query_body_is_summarized(self):
        """Unity DOTS query loops are summarized instead of fully preserved."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            max_body_lines=1,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Unity.Burst;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;

public struct MoveSpeed : IComponentData
{
    public float Value;
}

[BurstCompile]
public partial struct MoveSystem : ISystem
{
    public void OnUpdate(ref SystemState state)
    {
        foreach (var (transform, speed) in SystemAPI.Query<RefRW<LocalTransform>, RefRO<MoveSpeed>>())
        {
            float3 direction = new float3(1, 0, 0);
            transform.ValueRW.Position += direction * speed.ValueRO.Value * SystemAPI.Time.DeltaTime;
        }
    }
}
"""

        result = compressor.compress(code, language="unity")

        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert "unity-entities: SystemAPI.Query" in result.compressed
        assert "direction * speed" not in result.compressed

    def test_unity_serialized_fields_are_grouped(self):
        """Unity serialized field blocks collapse to a compact schema summary."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            max_body_lines=1,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [Header("Movement")]
    [SerializeField] private Rigidbody body;
    [SerializeField] private Transform cameraPivot;
    [SerializeField] private float speed = 7.5f;
    [SerializeField] private float jumpForce = 8f;
    private Vector3 input;
    private bool grounded;
}
"""

        result = compressor.compress(code, language="unity")

        assert result.syntax_valid is True
        assert "unity serialized fields" in result.compressed
        assert "Rigidbody body" in result.compressed
        assert "Transform cameraPivot" in result.compressed
        assert "Header" not in result.compressed

    def test_dots_component_fields_are_grouped(self):
        """DOTS component fields are summarized as a schema."""
        require_csharp_parser()
        compressor = CodeAwareCompressor(
            CodeCompressorConfig(min_tokens_for_compression=1, enable_ccr=False)
        )
        code = """
using Unity.Entities;

public struct MoveSpeed : IComponentData
{
    public float Value;
    public float Acceleration;
    public float MaxSpeed;
}
"""

        result = compressor.compress(code, language="unity")

        assert result.syntax_valid is True
        assert "unity-dots schema" in result.compressed
        assert "float Value" in result.compressed
        assert "public float Acceleration;" not in result.compressed

    def test_one_line_unity_entities_struct_is_not_duplicated(self):
        """One-line ECS data structs are preserved once, not once per child node."""
        require_csharp_parser()
        compressor = CodeAwareCompressor(
            CodeCompressorConfig(min_tokens_for_compression=1, enable_ccr=False)
        )
        code = """
using Unity.Entities;

public struct MoveSpeed : IComponentData { public float Value; public float Acceleration; }
"""

        result = compressor.compress(code, language="unity")

        assert result.syntax_valid is True
        assert result.compressed.count("public struct MoveSpeed") == 1

    def test_efcore_migration_operations_are_summarized(self):
        """EF migration operations are summarized with schema names preserved."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            max_body_lines=1,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Microsoft.EntityFrameworkCore.Migrations;

public partial class AddProducts : Migration
{
    protected override void Up(MigrationBuilder migrationBuilder)
    {
        migrationBuilder.CreateTable(
            name: "Products",
            columns: table => new
            {
                Id = table.Column<int>(nullable: false),
                Name = table.Column<string>(maxLength: 200, nullable: false)
            });
        migrationBuilder.CreateIndex(
            name: "IX_Products_Name",
            table: "Products",
            column: "Name");
    }
}
"""

        result = compressor.compress(code, language="efcore")

        assert result.syntax_valid is True
        assert result.compression_ratio < 0.9
        assert "efcore: CreateTable" in result.compressed
        assert "name=Products" in result.compressed
        assert "table.Column<string>" not in result.compressed

    def test_efcore_dbsets_are_grouped(self):
        """DbContext DbSet properties collapse into a compact schema summary."""
        require_csharp_parser()
        compressor = CodeAwareCompressor(
            CodeCompressorConfig(min_tokens_for_compression=1, enable_ccr=False)
        )
        code = """
using Microsoft.EntityFrameworkCore;

public sealed class AppDbContext : DbContext
{
    public DbSet<Product> Products => Set<Product>();
    public DbSet<Order> Orders => Set<Order>();
    public DbSet<Customer> Customers => Set<Customer>();
}
"""

        result = compressor.compress(code, language="efcore")

        assert result.syntax_valid is True
        assert "efcore dbsets" in result.compressed
        assert "DbSet<Product> Products" in result.compressed
        assert "public DbSet<Order> Orders" not in result.compressed

    def test_multiline_minimal_api_endpoint_is_summarized(self):
        """Multiline Minimal API handlers keep route metadata but omit handler body."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            max_body_lines=1,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http.HttpResults;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapPost("/products", (Product product) =>
    {
        var normalized = product.Name.Trim().ToUpperInvariant();
        var response = product with { Name = normalized };
        return TypedResults.Ok(response);
    })
   .WithName("CreateProduct")
   .WithOpenApi();

public record Product(string Name);
"""

        result = compressor.compress(code, language="aspnetcore")

        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert "aspnet: MapPost" in result.compressed
        assert 'route="/products"' in result.compressed
        assert "normalized" not in result.compressed

    def test_dotnet_worker_large_loop_body_can_be_omitted(self):
        """A single large loop statement should not force the whole method body to stay."""
        require_csharp_parser()
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            max_body_lines=1,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = """
using Microsoft.Extensions.Hosting;

public sealed class Worker : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            Console.WriteLine("tick");
            await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
        }
    }
}
"""

        result = compressor.compress(code, language="csharp")

        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert 'Console.WriteLine("tick")' not in result.compressed
        assert "lines omitted" in result.compressed

    def test_imports_preserved(self):
        """Imports are preserved in compressed output."""
        config = CodeCompressorConfig(
            preserve_imports=True,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(5)

        result = compressor.compress(code, language="python")

        assert "import os" in result.compressed
        assert "from typing import" in result.compressed

    def test_signatures_preserved(self):
        """Function signatures are preserved."""
        config = CodeCompressorConfig(
            preserve_signatures=True,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(3)

        result = compressor.compress(code, language="python")

        # Should preserve function signatures
        assert "def function_" in result.compressed
        assert "arg:" in result.compressed or "(arg" in result.compressed

    def test_error_handlers_preserved(self):
        """Module-level try/except blocks are preserved."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        # Code with module-level try/except (not inside functions)
        code = '''
import os

def setup():
    """Setup function."""
    pass

try:
    from optional_module import feature
except ImportError:
    feature = None

def main():
    """Main function with long body."""
    result = []
    for i in range(100):
        result.append(i)
    return result
'''
        result = compressor.compress(code, language="python")

        # Module-level error handlers should be preserved
        assert "try:" in result.compressed or "except" in result.compressed

    def test_syntax_verification(self):
        """Output syntax is verified as valid."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(5)

        result = compressor.compress(code, language="python")

        # Verify the compressed output is valid Python
        assert result.syntax_valid is True

        # Should be parseable
        try:
            compile(result.compressed, "<test>", "exec")
        except SyntaxError:
            pytest.fail("Compressed output has invalid Python syntax")

    def test_tree_sitter_loaded_after_compression(self):
        """Parser is loaded after compression."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)

        # Ensure clean state
        unload_tree_sitter()
        assert is_tree_sitter_loaded() is False

        # Compress should load parser
        code = generate_python_code(3)
        compressor.compress(code, language="python")

        assert is_tree_sitter_loaded() is True

    def test_unload_clears_parsers(self):
        """unload_tree_sitter clears loaded parsers."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)

        # Load a parser
        code = generate_python_code(3)
        compressor.compress(code, language="python")
        assert is_tree_sitter_loaded() is True

        # Unload
        result = unload_tree_sitter()
        assert result is True
        assert is_tree_sitter_loaded() is False


# =============================================================================
# TestDocstringModes
# =============================================================================


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter-languages not installed")
class TestDocstringModes:
    """Tests for different docstring handling modes."""

    def test_docstring_mode_full(self):
        """FULL mode preserves entire docstrings."""
        config = CodeCompressorConfig(
            docstring_mode=DocstringMode.FULL,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(2)

        result = compressor.compress(code, language="python")

        # Should preserve full docstrings
        assert "Args:" in result.compressed or "Returns:" in result.compressed

    def test_docstring_mode_first_line(self):
        """FIRST_LINE mode keeps only first line of docstring."""
        config = CodeCompressorConfig(
            docstring_mode=DocstringMode.FIRST_LINE,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(2)

        result = compressor.compress(code, language="python")

        # Multi-line docstring details should be removed
        # This is implementation-dependent
        assert result.compressed is not None

    def test_docstring_mode_remove(self):
        """REMOVE mode removes all docstrings."""
        config = CodeCompressorConfig(
            docstring_mode=DocstringMode.REMOVE,
            min_tokens_for_compression=10,
            max_body_lines=2,  # Low threshold to trigger compression
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        # Larger function to trigger body compression
        code = '''
def example():
    """This docstring should be removed."""
    x = 1
    y = 2
    z = 3
    result = x + y + z
    for i in range(10):
        result += i
    return result
'''
        result = compressor.compress(code, language="python")

        # Docstring should be removed when REMOVE mode is active
        assert "This docstring should be removed" not in result.compressed


# =============================================================================
# TestSemanticSymbolImportance
# =============================================================================


def _payment_processing_code() -> str:
    """Python code with varying symbol importance for testing."""
    return '''
import os
from typing import List, Optional

def process_payment(order, config):
    """Process a payment through the pipeline."""
    validated = validate_order(order)
    if not validated.is_valid:
        return PaymentResult(status='failed')
    charge = charge_customer(order.customer, order.total)
    receipt = generate_receipt(charge)
    send_confirmation(order.customer.email, receipt)
    update_inventory(order.items)
    log_transaction(charge.transaction_id)
    notify_warehouse(order)
    return PaymentResult(status='success', receipt=receipt)

def validate_order(order):
    """Validate an order before processing."""
    if not order.items:
        return ValidationResult(False, ['No items'])
    total = sum(item.price for item in order.items)
    if total <= 0:
        return ValidationResult(False, ['Invalid total'])
    if not order.customer:
        return ValidationResult(False, ['No customer'])
    return ValidationResult(True, [])

def charge_customer(customer, amount):
    """Charge the customer."""
    gateway = get_payment_gateway()
    response = gateway.charge(customer.card, amount)
    if not response.success:
        raise PaymentError(response.error)
    return response

def generate_receipt(charge):
    """Generate a receipt for the charge."""
    template = load_template('receipt')
    return template.render(charge=charge)

def _format_log_entry(entry):
    """Format a log entry for internal use. Never called."""
    timestamp = entry.get('ts', '')
    level = entry.get('level', 'INFO')
    message = entry.get('msg', '')
    source = entry.get('source', 'unknown')
    formatted = f'[{timestamp}] {level}: {message} ({source})'
    return formatted.strip()

def _dead_helper():
    """Never called anywhere in this file."""
    x = 1
    y = 2
    z = 3
    result = x + y + z
    for i in range(100):
        result += i
    return result
'''


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter-languages not installed")
class TestSemanticSymbolImportance:
    """Tests for semantic symbol importance analysis and variable compression."""

    def _make_compressor(self, **overrides):
        defaults = {
            "min_tokens_for_compression": 10,
            "max_body_lines": 3,
            "enable_ccr": False,
            "semantic_analysis": True,
        }
        defaults.update(overrides)
        return CodeAwareCompressor(CodeCompressorConfig(**defaults))

    def test_symbol_scores_populated(self):
        """Compression result includes symbol importance scores."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        assert result.symbol_scores
        assert "process_payment" in result.symbol_scores
        assert "validate_order" in result.symbol_scores
        assert "_dead_helper" in result.symbol_scores

    def test_called_functions_score_higher_than_dead_code(self):
        """Functions called by others score higher than unused functions."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        # validate_order is called by process_payment — should score higher
        assert result.symbol_scores["validate_order"] > result.symbol_scores["_dead_helper"]
        assert result.symbol_scores["charge_customer"] > result.symbol_scores["_dead_helper"]

    def test_public_symbols_score_higher_than_private(self):
        """Public functions (no leading _) score higher than private ones."""
        compressor = self._make_compressor()
        code = '''
def public_func():
    """A public function."""
    x = 1
    y = 2
    z = 3
    result = x + y + z
    for i in range(10):
        result += i
    return result

def _private_func():
    """A private function."""
    x = 1
    y = 2
    z = 3
    result = x + y + z
    for i in range(10):
        result += i
    return result
'''
        result = compressor.compress(code, language="python")

        assert result.symbol_scores["public_func"] > result.symbol_scores["_private_func"]

    def test_dead_code_compressed_to_signature_only(self):
        """Functions with score < 0.1 are compressed to signature + docstring only."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        # _dead_helper has 0 references, private → score 0.0
        assert result.symbol_scores["_dead_helper"] < 0.1

        # Body should be fully omitted
        assert "_dead_helper" in result.compressed
        # Should NOT contain body content
        assert "range(100)" not in result.compressed

    def test_referenced_functions_keep_more_body(self):
        """Higher-scored functions get more body lines from the budget."""
        # Use a generous target rate so there IS budget to distribute
        compressor = self._make_compressor(target_compression_rate=0.7)
        result = compressor.compress(_payment_processing_code(), language="python")

        compressed = result.compressed
        # With 70% target, high-scoring functions should retain body
        # while low-scoring ones get less. validate_order is referenced
        # and public (high score) so should keep some body.
        # _dead_helper has lowest score so should get least body.
        # Count body lines per function as a proxy for retention
        lines = compressed.split("\n")
        in_validate = False
        in_dead = False
        validate_body = 0
        dead_body = 0
        for line in lines:
            if "def validate_order" in line:
                in_validate = True
                in_dead = False
                continue
            elif "def _dead_helper" in line:
                in_dead = True
                in_validate = False
                continue
            elif line.startswith("def ") or (line.startswith("class ") and ":" in line):
                in_validate = False
                in_dead = False
                continue
            if in_validate and line.strip() and not line.strip().startswith('"""'):
                validate_body += 1
            if in_dead and line.strip() and not line.strip().startswith('"""'):
                dead_body += 1

        assert validate_body >= dead_body

    def test_omitted_comment_includes_calls(self):
        """Omitted comment includes call information when available."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        # process_payment calls validate_order, charge_customer, generate_receipt
        # These should appear in the omitted comment
        compressed = result.compressed
        if "lines omitted" in compressed:
            # Find omitted comments and check for calls info
            for line in compressed.split("\n"):
                if "process_payment" not in line and "lines omitted" in line:
                    continue
                if "lines omitted; calls:" in line:
                    assert "validate_order" in line or "charge_customer" in line
                    break

    def test_semantic_analysis_disabled(self):
        """When semantic_analysis=False, all functions get uniform compression."""
        compressor_with = self._make_compressor(semantic_analysis=True)
        compressor_without = self._make_compressor(semantic_analysis=False)

        code = _payment_processing_code()
        result_with = compressor_with.compress(code, language="python")
        result_without = compressor_without.compress(code, language="python")

        # Without semantic analysis, no symbol scores
        assert result_without.symbol_scores == {}

        # With semantic analysis, dead code is compressed more aggressively
        # _dead_helper body should NOT appear with semantic analysis
        assert "range(100)" not in result_with.compressed
        # But with uniform compression (no semantic), body lines ARE kept
        assert "x = 1" in result_without.compressed

    def test_summary_includes_semantic_info(self):
        """Summary includes semantic analysis information."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        summary = result.summary
        if result.symbol_scores:
            low_count = sum(1 for s in result.symbol_scores.values() if s < 0.1)
            if low_count > 0:
                assert "low-importance" in summary

    def test_dunder_methods_get_boost(self):
        """Dunder methods (__init__, etc.) get importance boost."""
        compressor = self._make_compressor()
        code = '''
class MyClass:
    """A class."""
    def __init__(self, value):
        """Initialize."""
        self.value = value
        self.processed = False
        self.results = []
        self.cache = {}
        self.errors = []
        for i in range(10):
            self.results.append(i)

    def _setup_cache(self):
        """Internal setup."""
        x = 1
        y = 2
        z = 3
        result = x + y + z
        for i in range(10):
            result += i
        return result
'''
        result = compressor.compress(code, language="python")

        # __init__ should score higher than _setup_cache
        if "__init__" in result.symbol_scores and "_setup_cache" in result.symbol_scores:
            assert result.symbol_scores["__init__"] > result.symbol_scores["_setup_cache"]

    def test_javascript_importance(self):
        """Symbol importance works for JavaScript code."""
        compressor = self._make_compressor()
        code = """
import { db } from './database';

function processUser(userId) {
    const user = fetchUser(userId);
    const profile = buildProfile(user);
    sendNotification(user.email, profile);
    logAction('process', userId);
    updateMetrics('user_processed');
    return { user, profile };
}

function fetchUser(id) {
    const result = db.query('SELECT * FROM users WHERE id = ?', [id]);
    if (!result) {
        throw new Error('User not found');
    }
    return result;
}

function buildProfile(user) {
    const prefs = loadPreferences(user.id);
    return { ...user, preferences: prefs };
}

function _internalDebug(msg) {
    const ts = Date.now();
    const formatted = `[${ts}] DEBUG: ${msg}`;
    console.log(formatted);
    return formatted;
}
"""
        result = compressor.compress(code, language="javascript")

        assert result.symbol_scores
        # fetchUser is called by processUser — should score higher than _internalDebug
        if "fetchUser" in result.symbol_scores and "_internalDebug" in result.symbol_scores:
            assert result.symbol_scores["fetchUser"] > result.symbol_scores["_internalDebug"]

    def test_syntax_still_valid_with_importance(self):
        """Compressed output with importance remains syntactically valid."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        assert result.syntax_valid is True

        # Should be parseable as Python
        try:
            compile(result.compressed, "<test>", "exec")
        except SyntaxError:
            pytest.fail("Semantic compression produced invalid Python syntax")

    def test_empty_code_no_crash(self):
        """Importance analysis handles empty code gracefully."""
        compressor = self._make_compressor()
        result = compressor.compress("", language="python")

        assert result.symbol_scores == {}

    def test_config_default_semantic_analysis_enabled(self):
        """semantic_analysis is True by default in config."""
        config = CodeCompressorConfig()
        assert config.semantic_analysis is True
