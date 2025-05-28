"use client";

import type React from "react";

import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Loader2,
  Search,
  ExternalLink,
  Tag,
  AlertCircle,
  CheckCircle,
  Info,
} from "lucide-react";
import { Separator } from "@/components/ui/separator";

interface Recommendation {
  api_id: number;
  name: string;
  description: string;
  url: string;
  tags: string[];
  explanation: string;
  scores: {
    final_score: number;
    embedding_score: number;
    tag_bonus: number;
    tag_overlap: number;
    tag_coverage: number;
  };
  matching_capabilities: string[];
  additional_capabilities: string[];
}

interface ValidationInfo {
  valid: boolean;
  coverage: number;
  known_tags: string[];
  unknown_tags: string[];
  suggestions: string[];
  warnings: string[];
}

interface ApiResponse {
  recommendations: Recommendation[];
  validation: ValidationInfo;
  request_info: {
    tags: string;
    description: string;
    top_k: number;
    explainability_mode: boolean;
  };
  status: string;
  count: number;
  message?: string;
}

interface ValidTagsResponse {
  valid_tags: string[];
  statistics: {
    total_unique_tags: number;
    mashup_unique_tags: number;
    api_unique_tags: number;
    common_tags: number;
  };
  tag_frequencies: { [key: string]: number };
  status: string;
}

export default function ApiRecommenderPage() {
  const [tags, setTags] = useState("");
  const [description, setDescription] = useState("");
  const [topK, setTopK] = useState([5]);
  const [explainabilityMode, setExplainabilityMode] = useState(true);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverUrl, setServerUrl] = useState("http://localhost:5000");

  // New state for tag validation and suggestions
  const [validTags, setValidTags] = useState<string[]>([]);
  const [tagSuggestions, setTagSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [validTagsLoading, setValidTagsLoading] = useState(false);
  const [tagInputValue, setTagInputValue] = useState("");

  // Fetch valid tags on component mount
  useEffect(() => {
    const fetchValidTags = async () => {
      setValidTagsLoading(true);
      try {
        const response = await fetch(`${serverUrl}/valid-tags`);
        if (response.ok) {
          const data: ValidTagsResponse = await response.json();
          setValidTags(data.valid_tags);
        }
      } catch (err) {
        console.error("Failed to fetch valid tags:", err);
      } finally {
        setValidTagsLoading(false);
      }
    };

    fetchValidTags();
  }, [serverUrl]);

  // Helper function to get tag suggestions based on input
  const getTagSuggestions = (input: string): string[] => {
    if (!input.trim() || validTags.length === 0) return [];

    const inputLower = input.toLowerCase().trim();
    const suggestions = validTags
      .filter((tag) => tag.toLowerCase().includes(inputLower))
      .slice(0, 10); // Limit to 10 suggestions

    return suggestions;
  };

  // Helper function to validate if tags are valid
  const validateTagsInput = (
    tagsString: string
  ): { validTags: string[]; invalidTags: string[] } => {
    const inputTags = tagsString
      .split(",")
      .map((tag) => tag.trim().toLowerCase())
      .filter((tag) => tag.length > 0);
    const validTagsLower = validTags.map((tag) => tag.toLowerCase());

    const validInputTags: string[] = [];
    const invalidInputTags: string[] = [];

    inputTags.forEach((tag) => {
      if (validTagsLower.includes(tag)) {
        validInputTags.push(tag);
      } else {
        invalidInputTags.push(tag);
      }
    });

    return { validTags: validInputTags, invalidTags: invalidInputTags };
  };

  // Handle tag input changes with real-time suggestions
  const handleTagInputChange = (value: string) => {
    setTags(value);
    setTagInputValue(value);

    // Get suggestions for the current word being typed
    const words = value.split(",");
    const currentWord = words[words.length - 1].trim();

    if (currentWord.length > 0) {
      const suggestions = getTagSuggestions(currentWord);
      setTagSuggestions(suggestions);
      setShowSuggestions(suggestions.length > 0);
    } else {
      setShowSuggestions(false);
    }
  };

  // Handle suggestion selection
  const handleSuggestionSelect = (suggestion: string) => {
    const words = tags.split(",");
    words[words.length - 1] = suggestion;
    const newTags =
      words.join(", ") +
      (words.length > 1 || words[0] !== suggestion ? ", " : "");
    setTags(newTags);
    setTagInputValue(newTags);
    setShowSuggestions(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!tags.trim()) return;

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${serverUrl}/recommend`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          tags: tags.trim(),
          description: description.trim(),
          top_k: topK[0],
          explainability_mode: explainabilityMode,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const validateTags = async () => {
    if (!tags.trim()) return;

    try {
      const response = await fetch(`${serverUrl}/validate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          tags: tags.trim(),
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // You could show validation results in a separate state if needed
        console.log("Validation results:", data);
      }
    } catch (err) {
      console.error("Validation error:", err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900">API Recommender</h1>
          <p className="text-lg text-gray-600">
            Discover the perfect APIs for your project using Graph Neural
            Networks
          </p>
        </div>

        {/* Server Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Server Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <Label htmlFor="server-url" className="text-sm">
                Server URL:
              </Label>
              <Input
                id="server-url"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                placeholder="http://localhost:5000"
                className="max-w-xs"
              />
            </div>
          </CardContent>
        </Card>

        {/* Search Form */}
        <Card>
          <CardHeader>
            <CardTitle>Find API Recommendations</CardTitle>
            <CardDescription>
              Enter tags and an optional description to get personalized API
              recommendations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2 relative">
                <Label htmlFor="tags">Tags (required)</Label>
                <div className="relative">
                  <Input
                    id="tags"
                    value={tags}
                    onChange={(e) => handleTagInputChange(e.target.value)}
                    onFocus={() => {
                      if (tagSuggestions.length > 0) setShowSuggestions(true);
                    }}
                    onBlur={() => {
                      // Delay hiding suggestions to allow clicks
                      setTimeout(() => setShowSuggestions(false), 200);
                    }}
                    placeholder="e.g., social, mapping, location, authentication"
                    required
                    className={`${validTagsLoading ? "pr-8" : ""}`}
                  />
                  {validTagsLoading && (
                    <Loader2 className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 animate-spin text-gray-400" />
                  )}

                  {/* Tag Suggestions Dropdown */}
                  {showSuggestions && tagSuggestions.length > 0 && (
                    <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-md shadow-lg max-h-48 overflow-y-auto">
                      {tagSuggestions.map((suggestion, index) => (
                        <button
                          key={index}
                          type="button"
                          onClick={() => handleSuggestionSelect(suggestion)}
                          className="w-full text-left px-3 py-2 hover:bg-gray-100 focus:bg-gray-100 focus:outline-none text-sm"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* Tag Validation Feedback */}
                {tags.trim() && validTags.length > 0 && (
                  <div className="space-y-1">
                    {(() => {
                      const {
                        validTags: validInputTags,
                        invalidTags: invalidInputTags,
                      } = validateTagsInput(tags);
                      return (
                        <>
                          {validInputTags.length > 0 && (
                            <div className="flex items-center gap-1 text-sm">
                              <CheckCircle className="h-3 w-3 text-green-500" />
                              <span className="text-green-600">
                                Valid tags:
                              </span>
                              <div className="flex flex-wrap gap-1">
                                {validInputTags
                                  .slice(0, 5)
                                  .map((tag, index) => (
                                    <Badge
                                      key={index}
                                      variant="secondary"
                                      className="text-xs bg-green-100 text-green-700"
                                    >
                                      {tag}
                                    </Badge>
                                  ))}
                                {validInputTags.length > 5 && (
                                  <span className="text-xs text-green-600">
                                    +{validInputTags.length - 5} more
                                  </span>
                                )}
                              </div>
                            </div>
                          )}
                          {invalidInputTags.length > 0 && (
                            <div className="flex items-center gap-1 text-sm">
                              <AlertCircle className="h-3 w-3 text-red-500" />
                              <span className="text-red-600">
                                Invalid tags:
                              </span>
                              <div className="flex flex-wrap gap-1">
                                {invalidInputTags
                                  .slice(0, 5)
                                  .map((tag, index) => (
                                    <Badge
                                      key={index}
                                      variant="destructive"
                                      className="text-xs"
                                    >
                                      {tag}
                                    </Badge>
                                  ))}
                                {invalidInputTags.length > 5 && (
                                  <span className="text-xs text-red-600">
                                    +{invalidInputTags.length - 5} more
                                  </span>
                                )}
                              </div>
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </div>
                )}

                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-500">
                    Enter comma-separated tags that describe your project needs
                  </p>
                  {validTags.length > 0 && (
                    <p className="text-xs text-gray-400">
                      {validTags.length} valid tags available
                    </p>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="description">Description (optional)</Label>
                <Textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe your project or specific requirements..."
                  rows={3}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Number of Recommendations: {topK[0]}</Label>
                  <Slider
                    value={topK}
                    onValueChange={setTopK}
                    max={20}
                    min={1}
                    step={1}
                    className="w-full"
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    id="explainability"
                    checked={explainabilityMode}
                    onCheckedChange={setExplainabilityMode}
                  />
                  <Label htmlFor="explainability">Enable explanations</Label>
                </div>
              </div>

              <div className="flex space-x-2">
                <Button type="submit" disabled={loading || !tags.trim()}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Getting Recommendations...
                    </>
                  ) : (
                    <>
                      <Search className="mr-2 h-4 w-4" />
                      Get Recommendations
                    </>
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Error: {error}. Make sure your Flask server is running on{" "}
              {serverUrl}
            </AlertDescription>
          </Alert>
        )}

        {/* Results */}
        {results && (
          <div className="space-y-6">
            {/* Validation Info */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  {results.validation.valid ? (
                    <CheckCircle className="h-5 w-5 text-green-500" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-yellow-500" />
                  )}
                  <span>Tag Validation</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-700">
                      Coverage: {(results.validation.coverage * 100).toFixed(1)}
                      %
                    </p>
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{
                          width: `${results.validation.coverage * 100}%`,
                        }}
                      ></div>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-700">
                      Status:{" "}
                      {results.validation.valid ? "Valid" : "Needs Attention"}
                    </p>
                  </div>
                </div>

                {results.validation.known_tags.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-green-700 mb-2">
                      Known Tags:
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {results.validation.known_tags.map((tag, index) => (
                        <Badge
                          key={index}
                          variant="secondary"
                          className="bg-green-100 text-green-800"
                        >
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {results.validation.unknown_tags.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-red-700 mb-2">
                      Unknown Tags:
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {results.validation.unknown_tags.map((tag, index) => (
                        <Badge key={index} variant="destructive">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {results.validation.suggestions.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-blue-700 mb-2">
                      Suggestions:
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {results.validation.suggestions.map(
                        (suggestion, index) => (
                          <Badge
                            key={index}
                            variant="outline"
                            className="border-blue-300 text-blue-700"
                          >
                            {suggestion}
                          </Badge>
                        )
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Recommendations */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold">
                  Recommendations ({results.count})
                </h2>
                {results.message && (
                  <Alert className="max-w-md">
                    <Info className="h-4 w-4" />
                    <AlertDescription>{results.message}</AlertDescription>
                  </Alert>
                )}
              </div>

              {results.recommendations.map((rec, index) => (
                <Card
                  key={rec.api_id}
                  className="hover:shadow-lg transition-shadow"
                >
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="flex items-center space-x-2">
                          <span>
                            #{index + 1} {rec.name}
                          </span>
                          <Badge variant="outline">ID: {rec.api_id}</Badge>
                        </CardTitle>
                        <CardDescription className="mt-2">
                          {rec.description}
                        </CardDescription>
                      </div>
                      {rec.url && (
                        <Button variant="outline" size="sm" asChild>
                          <a
                            href={rec.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            <ExternalLink className="h-4 w-4 mr-1" />
                            Visit
                          </a>
                        </Button>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Scores */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                      <div>
                        <p className="font-medium">Final Score</p>
                        <p className="text-2xl font-bold text-blue-600">
                          {rec.scores.final_score.toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium">Embedding</p>
                        <p className="text-lg">
                          {rec.scores.embedding_score.toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium">Tag Bonus</p>
                        <p className="text-lg">
                          {rec.scores.tag_bonus.toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium">Tag Overlap</p>
                        <p className="text-lg">{rec.scores.tag_overlap}</p>
                      </div>
                      <div>
                        <p className="font-medium">Coverage</p>
                        <p className="text-lg">
                          {(rec.scores.tag_coverage * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    <Separator />

                    {/* Tags and Capabilities */}
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm font-medium text-gray-700 mb-2">
                          API Tags:
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {rec.tags.map((tag, tagIndex) => (
                            <Badge key={tagIndex} variant="secondary">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {rec.matching_capabilities.length > 0 && (
                        <div>
                          <p className="text-sm font-medium text-green-700 mb-2">
                            Matching Capabilities:
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {rec.matching_capabilities.map((cap, capIndex) => (
                              <Badge
                                key={capIndex}
                                className="bg-green-100 text-green-800"
                              >
                                {cap}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {rec.additional_capabilities.length > 0 && (
                        <div>
                          <p className="text-sm font-medium text-blue-700 mb-2">
                            Additional Capabilities:
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {rec.additional_capabilities.map(
                              (cap, capIndex) => (
                                <Badge
                                  key={capIndex}
                                  variant="outline"
                                  className="border-blue-300 text-blue-700"
                                >
                                  {cap}
                                </Badge>
                              )
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Explanation */}
                    {rec.explanation && explainabilityMode && (
                      <div>
                        <p className="text-sm font-medium text-gray-700 mb-2">
                          Explanation:
                        </p>
                        <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded-md">
                          {rec.explanation}
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
