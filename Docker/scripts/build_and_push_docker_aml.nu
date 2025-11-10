#!/usr/bin/env nu

# Build and push Docker image to Azure Container Registry
# Usage: nu build_and_push_docker_aml.nu <registry-name> [tag]

const path_to_script = (path self)

def main [registry_name: string = "atupinirgacr", tag: string = "latest"] {
    let image_name = "embodiedbenchrunner-without-install"
    let script_dir = ($path_to_script | path dirname)
    let dockerfile_path = ($script_dir | path join ".." "Dockerfile.aml")
    let build_context = ($script_dir | path join ".." "..")
    let registry_url = $"($registry_name).azurecr.io"
    let full_image_name = $"($registry_url)/($image_name):($tag)"
    
    print $"Building Docker image: ($image_name):($tag)"
    print $"Target registry: ($registry_url)"
    print ""
    
    # Build the Docker image
    print "Building Docker image..."
    ^docker build -f $dockerfile_path -t $"($image_name):($tag)" $build_context
    
    if $env.LAST_EXIT_CODE != 0 {
        print "Error: Docker build failed"
        exit 1
    }
    
    print $"Successfully built ($image_name):($tag)"
    print ""
    
    # Login to Azure Container Registry
    print "Logging in to Azure Container Registry..."
    ^az acr login -n $registry_name
    
    if $env.LAST_EXIT_CODE != 0 {
        print "Error: ACR login failed"
        exit 1
    }
    
    print "Successfully logged in"
    print ""
    
    # Tag the image for the registry
    print $"Tagging image as ($full_image_name)"
    ^docker tag $"($image_name):($tag)" $full_image_name
    
    if $env.LAST_EXIT_CODE != 0 {
        print "Error: Docker tag failed"
        exit 1
    }
    
    print ""
    
    # Push the image to the registry
    print $"Pushing image to ($registry_url)..."
    ^docker push $full_image_name
    
    if $env.LAST_EXIT_CODE != 0 {
        print "Error: Docker push failed"
        exit 1
    }
    
    print ""
    print $"✓ Successfully pushed ($full_image_name)"
    print ""
    print "To use this image in your YAML file:"
    print "environment:"
    print $"  image: ($image_name):($tag)"
    print $"  registry: ($registry_url)"
}