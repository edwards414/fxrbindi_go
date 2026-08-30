#!/usr/bin/env ruby
# Add one processed App Store Connect build to the internal TestFlight group.

require "base64"
require "json"
require "net/http"
require "openssl"
require "uri"

def required_env(name)
  value = ENV[name]
  abort("Missing environment variable: #{name}") if value.nil? || value.empty?
  value
end

def base64url(value)
  Base64.urlsafe_encode64(value, padding: false)
end

def app_store_token
  header = base64url(JSON.generate(alg: "ES256", kid: required_env("ASC_KEY_ID"), typ: "JWT"))
  now = Time.now.to_i
  payload = base64url(JSON.generate(
    iss: required_env("ASC_ISSUER_ID"),
    iat: now,
    exp: now + 600,
    aud: "appstoreconnect-v1",
  ))
  signing_input = "#{header}.#{payload}"
  key = OpenSSL::PKey::EC.new(required_env("ASC_PRIVATE_KEY"))
  digest = OpenSSL::Digest::SHA256.digest(signing_input)
  sequence = OpenSSL::ASN1.decode(key.dsa_sign_asn1(digest))
  raw_signature = sequence.value.map { |integer| integer.value.to_s(2).rjust(32, "\0") }.join
  "#{signing_input}.#{base64url(raw_signature)}"
end

def request(method, path, token, query: nil, body: nil)
  uri = URI("https://api.appstoreconnect.apple.com#{path}")
  uri.query = URI.encode_www_form(query) if query
  klass = { get: Net::HTTP::Get, post: Net::HTTP::Post }.fetch(method)
  req = klass.new(uri)
  req["Authorization"] = "Bearer #{token}"
  req["Content-Type"] = "application/json"
  req.body = JSON.generate(body) if body
  response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: true) { |http| http.request(req) }
  return response if response.code.to_i.between?(200, 299)

  abort("App Store Connect #{method.upcase} #{path} failed (#{response.code}): #{response.body}")
end

build_number = ARGV.fetch(0) { abort("Usage: #{$PROGRAM_NAME} BUILD_NUMBER") }
token = app_store_token
builds = JSON.parse(request(
  :get,
  "/v1/builds",
  token,
  query: {
    "filter[app]" => required_env("ASC_APP_ID"),
    "filter[version]" => build_number,
    "sort" => "-uploadedDate",
    "limit" => "1",
  },
).body).fetch("data")
abort("Processed build #{build_number} was not found") if builds.empty?

build_id = builds.first.fetch("id")
group_id = required_env("ASC_BETA_GROUP_ID")
relationships = JSON.parse(request(
  :get,
  "/v1/betaGroups/#{group_id}/relationships/builds",
  token,
  query: { "limit" => "200" },
).body).fetch("data")

if relationships.any? { |item| item["id"] == build_id }
  puts "Build #{build_number} is already available to internal testers"
  exit 0
end

request(
  :post,
  "/v1/betaGroups/#{group_id}/relationships/builds",
  token,
  body: { data: [{ type: "builds", id: build_id }] },
)
puts "Build #{build_number} is now available to internal testers"
